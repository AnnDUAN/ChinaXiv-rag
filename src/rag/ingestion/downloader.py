"""从 ChinaXiv（中国科学院科技论文预发布平台）抓论文列表并下载 PDF。

页面流程（2026-10 实测）：
1. 先访问首页，拿到 JSESSIONID；
2. 打开 /user/search.htm，页面会生成一个 pageId；
3. 用 pageId + filterField=domain&value=243 筛选“计算机科学”；
4. 翻页用 pageId + setId=recordList&currentPage=N（N 从 0 开始）；
5. 每条搜索结果自带标题、作者、摘要、分类和 PDF 下载地址；
6. PDF 不需要登录，直接下载即可。

下载地址形如：
    https://chinaxiv.org/user/download.htm?uuid=<uuid>&filetype=pdf

默认只保留“标题是中文”的论文。判断方法不是看分类，而是看标题里
有没有足够多的汉字，因此“基于 BERT 的……”会保留，纯英文标题会被过滤。
"""

from __future__ import annotations

import html as html_lib
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from .. import config
from ..common.utils import safe_filename

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

BASE_URL = "https://chinaxiv.org"
SEARCH_URL = f"{BASE_URL}/user/search.htm"
CS_DOMAIN_ID = "243"  # ChinaXiv 的“计算机科学”分类编号
CS_SUBJECT_IDS = {
    "244",  # 计算机科学的集成理论
    "245",  # 计算机软件
    "359",  # 计算机体系结构
    "360",  # 计算机硬件技术
    "361",  # 计算机应用技术
    "362",  # 自然语言理解与机器翻译
    "363",  # 信息安全
    "364",  # 计算机网络
    "768",  # 计算机科学技术其他学科
}
PAGE_SIZE = 20  # 每页实际返回 20 条
CATALOG_FILE = config.DATA_DIR / "catalog.json"
IGNORED_FILE = config.DATA_DIR / "ignored.json"

CJK = re.compile(r"[一-鿿]")
PAPER_ID = re.compile(r"(\d{6}\.\d+)")


def pdf_path_for(paper_id: str, out_dir: Path | None = None) -> Path:
    """论文 PDF 的统一命名：data/pdfs/<论文编号>.pdf。"""
    out_dir = out_dir or config.PDF_DIR
    return out_dir / f"{safe_filename(paper_id)}.pdf"


def is_valid_pdf(path: Path) -> bool:
    """文件存在、非空，并且开头是 %PDF，才算已经下载好。"""
    try:
        if not path.exists() or path.stat().st_size < 8:
            return False
        with path.open("rb") as f:
            return f.read(4) == b"%PDF"
    except OSError:
        return False


def is_downloaded(paper_id: str, out_dir: Path | None = None) -> bool:
    return is_valid_pdf(pdf_path_for(paper_id, out_dir))


def _clean_html(fragment: str) -> str:
    """把一小段 HTML 变成干净文字。"""
    text = re.sub(r"<[^>]+>", " ", fragment)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _paper_id(value: str) -> str:
    """从论文编号或详情页链接里提取 202610.00065 这种编号。"""
    match = PAPER_ID.search(value)
    return match.group(1) if match else value.strip()


def is_chinese_title(title: str) -> bool:
    """标题里至少有 4 个汉字，才当作中文论文。

    “基于 BERT 的文本分类”这种中英混合标题会保留；
    “Side-Path Parametric Memory for LLM Agents”这种纯英文标题会被过滤。
    """
    return len(CJK.findall(title or "")) >= 4


def _new_session() -> requests.Session:
    """建立带 cookie 的会话，ChinaXiv 的检索页依赖它。"""
    session = requests.Session()
    session.headers.update(UA)
    resp = session.get(f"{BASE_URL}/", timeout=60)
    resp.raise_for_status()
    return session


def _search_page_id(session: requests.Session) -> str:
    """打开检索页，拿到页面自动生成的 pageId。"""
    resp = session.get(SEARCH_URL, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    match = re.search(r"pageId=(\d+)", resp.text)
    if not match:
        raise RuntimeError("ChinaXiv 检索页里没有找到 pageId，网站结构可能变了。")
    return match.group(1)


def _parse_result_page(text: str) -> list[dict[str, Any]]:
    """从检索结果页里提取论文条目。"""
    items: list[dict[str, Any]] = []
    for block in re.findall(r"<li[^>]*>(.*?)</li>", text, re.S):
        match = re.search(r'href="/abs/(\d{6}\.\d+)"', block)
        if not match:
            continue
        paper_id = match.group(1)

        title_match = re.search(r"<h3>\s*<a[^>]*>(.*?)</a>\s*</h3>", block, re.S)
        if not title_match:
            continue
        title = _clean_html(title_match.group(1))

        download_match = re.search(
            r'href="(/user/download\.htm\?uuid=[^"]+)"', block
        )
        download_url = (
            html_lib.unescape(download_match.group(1)) if download_match else ""
        )
        if download_url and "filetype=" not in download_url:
            download_url += "&filetype=pdf"
        if download_url.startswith("/"):
            download_url = BASE_URL + download_url

        authors: list[str] = []
        for name in re.findall(
            r'<a href="/user/search\.htm\?field=author&value=[^"]*"[^>]*>(.*?)</a>',
            block,
            re.S,
        ):
            name = _clean_html(name)
            if name and name not in authors:
                authors.append(name)

        # 一篇论文可能同时挂在多个分类下，优先显示其中的计算机类分类
        subjects = [
            (sid, _clean_html(name))
            for sid, name in re.findall(
                r'field=subject&value=(\d+)"[^>]*>(.*?)</a>', block, re.S
            )
        ]
        cs_subjects = [name for sid, name in subjects if sid in CS_SUBJECT_IDS]
        subject_detail = (
            cs_subjects[0]
            if cs_subjects
            else (subjects[0][1] if subjects else "")
        )

        abstract_match = re.search(r"摘要：</font>\s*(.*?)</p>", block, re.S)
        abstract = _clean_html(abstract_match.group(1)) if abstract_match else ""
        # 列表页的摘要后面会跟一个“展开 -->”的链接，去掉它
        abstract = re.sub(r"\s*(?:展开|收起)\s*(?:-->)?\s*$", "", abstract).strip()

        date_match = re.search(r"提交时间：</font>\s*([\d-]+)", block)

        items.append(
            {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "url": f"{BASE_URL}/abs/{paper_id}",
                "download_url": download_url,
                "subject": "计算机科学",
                "subject_detail": subject_detail,
                "subjects": [name for _, name in subjects],
                "submitted": date_match.group(1) if date_match else "",
                "source": "ChinaXiv",
                "language": "zh" if is_chinese_title(title) else "en",
            }
        )
    return items


def fetch_chinaxiv_catalog(
    pages: int = 1,
    domain: str = CS_DOMAIN_ID,
    chinese_only: bool = True,
    delay: float = 0.8,
) -> dict[str, Any]:
    """抓取 ChinaXiv 指定分类的论文列表。

    返回：
        {
          "items": 过滤后的论文列表,
          "fetched": 原始抓到的篇数,
          "filtered_english": 因英文标题被过滤的篇数,
          "pages": 实际抓取页数,
          "domain": 分类编号,
        }
    """
    pages = max(1, int(pages))
    session = _new_session()
    page_id = _search_page_id(session)

    raw_items: list[dict[str, Any]] = []
    for page_index in range(pages):
        if page_index == 0:
            url = (
                f"{SEARCH_URL}?pageId={page_id}"
                f"&type=filter&filterField=domain&value={domain}"
            )
        else:
            url = (
                f"{SEARCH_URL}?pageId={page_id}"
                f"&setId=recordList&currentPage={page_index}"
            )
        resp = session.get(url, headers={"Referer": SEARCH_URL}, timeout=60)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        page_items = _parse_result_page(resp.text)
        if not page_items:
            break
        raw_items.extend(page_items)
        if page_index + 1 < pages:
            time.sleep(delay)

    # 同一页里偶尔会有重复链接，按编号去重并保持原顺序
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in raw_items:
        if item["paper_id"] in seen:
            continue
        seen.add(item["paper_id"])
        unique.append(item)

    kept = (
        [item for item in unique if is_chinese_title(item["title"])]
        if chinese_only
        else unique
    )
    filtered_english = len(unique) - len(kept)

    # 已经明确不想要的论文，后续抓取时直接跳过
    ignored = load_ignored()
    before_ignored = len(kept)
    kept = [item for item in kept if item["paper_id"] not in ignored]
    filtered_ignored = before_ignored - len(kept)

    return {
        "items": kept,
        "fetched": len(unique),
        "filtered_english": filtered_english,
        "filtered_ignored": filtered_ignored,
        "pages": pages,
        "domain": domain,
    }


def resolve_chinaxiv_download_url(
    session: requests.Session, paper_id: str
) -> tuple[str, str]:
    """打开详情页，解析出 PDF 下载地址。

    返回 (下载地址, 详情页地址)。
    """
    paper_id = _paper_id(paper_id)
    detail_url = f"{BASE_URL}/abs/{paper_id}"
    resp = session.get(detail_url, timeout=60)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    match = re.search(
        r'href="(/user/download\.htm\?uuid=[^"]+?&filetype=pdf)"', resp.text
    )
    if not match:
        match = re.search(r'href="(/user/download\.htm\?uuid=[^"]+)"', resp.text)
    if not match:
        raise RuntimeError(
            "详情页里没有找到 PDF 下载地址，可能这篇论文没有公开全文。"
            f"详情页：{detail_url}"
        )
    url = html_lib.unescape(match.group(1))
    if "filetype=" not in url:
        url += "&filetype=pdf"
    if url.startswith("/"):
        url = BASE_URL + url
    return url, detail_url


def download_chinaxiv_pdf(
    session: requests.Session,
    paper_id: str,
    download_url: str | None = None,
    out_dir: Path | None = None,
    delay: float = 0.0,
) -> Path:
    """下载单篇 ChinaXiv 论文的 PDF。"""
    paper_id = _paper_id(paper_id)
    detail_url = f"{BASE_URL}/abs/{paper_id}"
    if not download_url:
        download_url, detail_url = resolve_chinaxiv_download_url(session, paper_id)
    if delay:
        time.sleep(delay)

    resp = session.get(
        download_url, headers={"Referer": detail_url}, timeout=120
    )
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if resp.content[:4] != b"%PDF":
        preview = resp.text[:300].replace("\n", " ")
        raise RuntimeError(
            f"这个地址返回的不是 PDF（{content_type}）：{preview}"
        )

    out_dir = out_dir or config.PDF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_path_for(paper_id, out_dir)
    path.write_bytes(resp.content)
    return path


def download_chinaxiv_papers(
    papers: Iterable[str | dict[str, Any]],
    out_dir: Path | None = None,
    delay: float = 1.5,
    force: bool = False,
) -> list[dict[str, Any]]:
    """批量下载 ChinaXiv 论文。

    papers 可以是论文编号列表，也可以是 catalog 里的字典（带 download_url）。
    已有有效 PDF 的默认跳过；force=True 才覆盖重下。
    返回 [{paper_id, path, status, error}]，status 为 downloaded / skipped / failed。
    """
    entries = list(papers)
    results: list[dict[str, Any]] = []
    if not entries:
        return results

    session: requests.Session | None = None
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            paper_id = _paper_id(str(entry.get("paper_id", "")))
            download_url = entry.get("download_url") or None
        else:
            paper_id = _paper_id(str(entry))
            download_url = None

        path = pdf_path_for(paper_id, out_dir)
        if not paper_id:
            results.append(
                {"paper_id": paper_id, "path": None, "status": "failed",
                 "error": "论文编号为空"}
            )
        elif not force and is_valid_pdf(path):
            results.append(
                {"paper_id": paper_id, "path": path, "status": "skipped",
                 "error": None}
            )
        else:
            if session is None:
                session = _new_session()
            errors: list[str] = []
            saved: Path | None = None
            # 先用列表页给的地址；失败再用详情页重新解析一次
            attempts = [download_url, None] if download_url else [None]
            for attempt_url in attempts:
                try:
                    saved = download_chinaxiv_pdf(
                        session,
                        paper_id,
                        attempt_url,
                        out_dir=out_dir,
                    )
                    break
                except Exception as exc:  # 网络、页面结构变化等
                    errors.append(str(exc))
            if saved is not None:
                results.append(
                    {"paper_id": paper_id, "path": saved, "status": "downloaded",
                     "error": None}
                )
            else:
                results.append(
                    {"paper_id": paper_id, "path": None, "status": "failed",
                     "error": "；".join(dict.fromkeys(errors))}
                )

        if index + 1 < len(entries):
            time.sleep(delay)
    return results


def download_catalog(
    catalog: list[dict[str, Any]],
    out_dir: Path | None = None,
    limit: int = 0,
    delay: float = 1.5,
    force: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """按论文列表下载所有缺失的 PDF。

    返回 (下载结果, 已有 PDF 跳过的篇数, 缺失 PDF 的总篇数)。
    """
    missing: list[dict[str, Any]] = []
    skipped = 0
    ignored = load_ignored()
    for item in catalog:
        paper_id = item.get("paper_id")
        if not paper_id or paper_id in ignored:
            continue
        if not force and is_downloaded(paper_id, out_dir):
            skipped += 1
        else:
            missing.append(item)
    selected = missing[:limit] if limit > 0 else missing
    results = download_chinaxiv_papers(
        selected, out_dir=out_dir, delay=delay, force=force
    )
    return results, skipped, len(missing)


def download_pdf(
    url: str,
    name: str | None = None,
    referer: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    """直接下载任意 PDF 直链（保留给 --url 用）。"""
    headers = dict(UA)
    if referer:
        headers["Referer"] = referer
    resp = requests.get(url, headers=headers, timeout=90)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if not (resp.content[:4] == b"%PDF" or "application/pdf" in content_type):
        preview = resp.text[:300].replace("\n", " ")
        raise RuntimeError(
            f"这个地址返回的不是 PDF（{content_type}）：{preview}"
        )
    out_dir = out_dir or config.PDF_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(name or url.rsplit("/", 1)[-1] or "paper")
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    path = out_dir / filename
    path.write_bytes(resp.content)
    return path


def load_ignored(path: Path | None = None) -> set[str]:
    """读取不想再抓的论文编号列表。"""
    path = path or IGNORED_FILE
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    if isinstance(data, dict):
        data = data.get("ignored", data.get("items", []))
    if not isinstance(data, list):
        return set()
    return {str(item).strip() for item in data if str(item).strip()}


def save_ignored(paper_ids: Iterable[str], path: Path | None = None) -> Path:
    path = path or IGNORED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    values = sorted({str(item).strip() for item in paper_ids if str(item).strip()})
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    """读取已有的论文列表，按 paper_id 去重，并保持原顺序。"""
    path = path or CATALOG_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        return []
    seen: set[str] = set()
    ignored = load_ignored()
    result: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paper_id")
        if not paper_id or paper_id in seen or paper_id in ignored:
            continue
        seen.add(paper_id)
        result.append(item)
    return result


def merge_catalog(
    existing: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """把新抓到的论文合并进已有列表，按 paper_id 去重。

    返回 (合并后的列表, 新增篇数, 更新篇数)。
    """
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in existing:
        paper_id = item.get("paper_id")
        if paper_id and paper_id not in by_id:
            by_id[paper_id] = dict(item)
            order.append(paper_id)

    added = 0
    updated = 0
    for item in new_items:
        paper_id = item.get("paper_id")
        if not paper_id:
            continue
        if paper_id not in by_id:
            by_id[paper_id] = dict(item)
            order.append(paper_id)
            added += 1
            continue
        merged = dict(by_id[paper_id])
        changed = False
        for key, value in item.items():
            if value not in (None, "", [], {}) and merged.get(key) != value:
                merged[key] = value
                changed = True
        by_id[paper_id] = merged
        if changed:
            updated += 1

    return [by_id[paper_id] for paper_id in order], added, updated


def merge_catalog_file(
    new_items: list[dict[str, Any]],
    path: Path | None = None,
) -> tuple[Path, list[dict[str, Any]], int, int]:
    """抓取结果和已有 catalog.json 合并后保存，避免重复条目。"""
    path = path or CATALOG_FILE
    existing = load_catalog(path)
    merged, added, updated = merge_catalog(existing, new_items)
    save_catalog(merged, path)
    return path, merged, added, updated


def save_catalog(items: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or CATALOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
