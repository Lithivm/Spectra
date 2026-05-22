# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        *collect_data_files('mutagen'),
        ('assets/logo.png', 'assets'),
    ],
    hiddenimports=[
        'pyloudnorm',
        'pyfftw',
        'pyfftw.interfaces',
        'pyfftw.interfaces.scipy_fft',
        'scipy.signal',
        'scipy.special.cython_special',
        'sklearn.utils._cython_blas',
        'soxr',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        '_tkinter',
        'tcl',
        'tk',
        'tensorflow',
        'torch',
        'sympy',
        'pytest',
        'doctest',
        'ensurepip',
        'venv',
        'lib2to3',
        'xmlrpc',
        'babel',
        'sphinx',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Spectra',
    icon=r'assets\logo.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['*.cp310-win_amd64.pyd', 'numpy*.dll', 'scipy*.dll'],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=['*.cp310-win_amd64.pyd', 'numpy*.dll', 'scipy*.dll'],
    name='Spectra',
)
