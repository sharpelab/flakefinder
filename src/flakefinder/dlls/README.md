# Leica SDK DLLs

This directory contains the Leica HardwareModel SDK DLLs. These are proprietary binaries and are **not committed to git**.

Used by both `flakefinder.driver` (legacy) and `flakefinder.leica` (new API).

## Required Files

Copy from the microscope PC or Leica SDK installation:

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

These DLLs come from the Leica Microsystems SDK installation. On the microscope PC, check:

```
C:\Program Files\Leica Microsystems\...
```

Or extract from the SDK zip in `sharpelab/leica_sdk/AHM_SDK_V2020.3.3.10693/`.

## Note

The driver will fail to import without these DLLs. This is expected behavior on non-microscope machines.
