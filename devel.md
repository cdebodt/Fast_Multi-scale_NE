## Package devel notes


Useful resource: [How to Publish an Open-Source Python Package to PyPI](https://realpython.com/pypi-publish-python-package/)

### Install from local repo:

```
pip install .
```

### Build distribution

```
python -m build
```

or

```
python -m build -n
```

However, when uploading, I get

```
$ twine upload -r testpypi dist/*
Uploading distributions to https://test.pypi.org/legacy/
Enter your username: lgatto
Enter your password:
Uploading fmsne-0.5.0-cp310-cp310-linux_x86_64.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 160.9/160.9 kB • 00:00 • 2.5 MB/s
WARNING  Error during upload. Retry with the --verbose option for more details.
ERROR    HTTPError: 400 Bad Request from https://test.pypi.org/legacy/
         Binary wheel 'fmsne-0.5.0-cp310-cp310-linux_x86_64.whl' has an unsupported platform tag 'linux_x86_64'.
```

Some background:
- https://github.com/pypa/auditwheel
- https://github.com/pypa/manylinux

```
$ auditwheel show dist/fmsne-0.5.0-cp310-cp310-linux_x86_64.whl

fmsne-0.5.0-cp310-cp310-linux_x86_64.whl is consistent with the
following platform tag: "manylinux_2_31_x86_64".

The wheel references external versioned symbols in these
system-provided shared libraries: libgcc_s.so.1 with versions
{'GCC_3.0'}, libstdc++.so.6 with versions {'CXXABI_1.3',
'GLIBCXX_3.4', 'CXXABI_1.3.9'}, libm.so.6 with versions {'GLIBC_2.29',
'GLIBC_2.2.5'}, libc.so.6 with versions {'GLIBC_2.3.4', 'GLIBC_2.14',
'GLIBC_2.4', 'GLIBC_2.2.5'}

This constrains the platform tag to "manylinux_2_31_x86_64". In order
to achieve a more compatible tag, you would need to recompile a new
wheel from source on a system with earlier versions of these
libraries, such as a recent manylinux image.
```

```
$ auditwheel repair dist/fmsne-0.5.0-cp310-cp310-linux_x86_64.whl
INFO:auditwheel.main_repair:Repairing fmsne-0.5.0-cp310-cp310-linux_x86_64.whl
usage: auditwheel [-h] [-V] [-v] command ...
auditwheel: error: cannot repair "dist/fmsne-0.5.0-cp310-cp310-linux_x86_64.whl" to "manylinux_2_5_x86_64" ABI because of the presence of too-recent versioned symbols. You'll need to compile the wheel on an older toolchain.
```

### Build source dist

See https://docs.python.org/3/distutils/sourcedist.html

```
python setup.py sdist
```

### Check with twine

```
twine check dist/*
```

### Upload to testpypi

```
$ twine upload -r testpypi dist/*
```

See
- https://test.pypi.org/project/fmsne/0.5.0/
- https://test.pypi.org/project/fmsne/

## Install from testpypi

```
pip install -i https://test.pypi.org/simple/ fmsne==0.5.0
## or
pip install -i https://test.pypi.org/simple/ fmsne
```

### Upload to pypi

```
$ twine upload  dist/*
Uploading distributions to https://upload.pypi.org/legacy/
Enter your username: lgatto
Enter your password:
Uploading fmsne-0.5.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.4/294.4 kB • 00:01 • 842.0 kB/s
```

View at
- https://pypi.org/project/fmsne/0.5.0/
- https://pypi.org/project/fmsne/

## Install from pypi

```
pip install fmsne
```
