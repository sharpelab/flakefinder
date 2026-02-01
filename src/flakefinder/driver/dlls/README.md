# Leica SDK DLLs

This directory should contain the Leica HardwareModel SDK DLLs. These are proprietary binaries and are **not committed to git**.

## Required Files

Copy from the microscope PC or original 2DMatGMM-System installation:

```
hwmodel2.dll
hwmodel2exucapi.dll
ahmcore.dll
ahmcorelocator.dll
ahmconfig.xml
bgapi2_ext.dll
bgapi2_genicam.dll
bgapi2_img.dll
bgapi2_gige.cti
bgapi2_usb.cti
bopfdrvctl.dll
cmserial2.dll
neoisp_cpp.dll
tracelogger.dll
ucBgapi.dll
valentine.dll
BO_GigEFilterDrv.dll
GCBase_MD_VC141_v3_4.dll
GenApi_MD_VC141_v3_4.dll
Log_MD_VC141_v3_4.dll
MathParser_MD_VC141_v3_4.dll
NodeMapData_MD_VC141_v3_4.dll
XmlParser_MD_VC141_v3_4.dll
log4cpp_MD_VC141_v3_4.dll
```

## Source

These DLLs come from the Leica Microsystems SDK installation. On the microscope PC, they should be in:

```
C:\Program Files\Leica Microsystems\...
```

Or copy from the original repo:

```
2DMatGMM-System/Drivers/Full_Microscope_Driver/dlls/
```

## Note

The driver will fail to import without these DLLs. This is expected behavior on non-microscope machines.
