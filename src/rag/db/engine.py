"""数据库连接。

使用 psycopg3 同步连接，不引入 ORM。
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from .. import config


@contextlib.contextmanager
def connect(
    dsn: str | None = None,
    *,
    use_vector: bool = True,
) -> Iterator[psycopg.Connection]:
    """打开一个连接，退出时自动提交或回滚。

    use_vector=False 用于第一次建库：这时 vector 扩展可能还不存在，
    不能提前注册 vector 类型。
    """
    conn = psycopg.connect(
        dsn or config.DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=5,
    )
    try:
        if use_vector:
            register_vector(conn)
        with conn:
            yield conn
    finally:
        conn.close()
