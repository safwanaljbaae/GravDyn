import sys
import pytest
from pathlib import Path
from io import StringIO

from gravdyn.shape_verification import Tee


class TestTee:
    def test_writes_to_both_stdout_and_file(self, tmp_path):
        log = tmp_path / "test.log"
        buf = StringIO()
        tee = Tee(buf, str(log))
        tee.write("hello world\n")
        tee.flush()
        assert buf.getvalue() == "hello world\n"
        assert Path(log).read_text() == "hello world\n"
        tee.close()

    def test_multiple_writes(self, tmp_path):
        log = tmp_path / "multi.log"
        buf = StringIO()
        tee = Tee(buf, str(log))
        tee.write("line1\n")
        tee.write("line2\n")
        tee.close()
        assert buf.getvalue() == "line1\nline2\n"
        assert Path(log).read_text() == "line1\nline2\n"

    def test_close_closes_file(self, tmp_path):
        log = tmp_path / "close.log"
        buf = StringIO()
        tee = Tee(buf, str(log))
        tee.write("data\n")
        tee.close()
        assert Path(log).read_text() == "data\n"

    def test_flush_flushes_both(self, tmp_path):
        log = tmp_path / "flush.log"
        buf = StringIO()
        tee = Tee(buf, str(log))
        tee.write("flush me\n")
        tee.flush()
        assert buf.getvalue() == "flush me\n"
        assert Path(log).read_text() == "flush me\n"
        tee.close()
