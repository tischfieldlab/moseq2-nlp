from moseq2_nlp.util import *
import unittest
import os
import subprocess


class UtilTesting(unittest.TestCase):
    def test_ensure_dir(self):
        dir_name = "./tmp"
        ensure_dir(dir_name)
        self.assertTrue(os.path.exists(dir_name))
        subprocess.call(f"rm -rf {dir_name}", shell=True)

    def test_write_read_yaml(self):
        dct = {"a": 0, "b": 1, "c": 2}
        fn = "./tmp.yaml"
        write_yaml(fn, dct)
        self.assertTrue(os.path.exists(fn))
        read_dct = read_yaml(fn)
        self.assertTrue(dct == read_dct)
        subprocess.call(f"rm {fn}", shell=True)


if __name__ == "__main__":
    unittest.main()
