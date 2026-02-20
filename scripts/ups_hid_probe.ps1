# Probe APC UPS HID device for available feature reports
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class HidApi {
    [DllImport("hid.dll")]
    public static extern bool HidD_GetPreparsedData(IntPtr HidDeviceObject, ref IntPtr PreparsedData);

    [DllImport("hid.dll")]
    public static extern bool HidD_FreePreparsedData(IntPtr PreparsedData);

    [DllImport("hid.dll")]
    public static extern int HidP_GetCaps(IntPtr PreparsedData, ref HIDP_CAPS Capabilities);

    [DllImport("hid.dll")]
    public static extern bool HidD_GetFeature(IntPtr HidDeviceObject, byte[] ReportBuffer, int ReportBufferLength);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern IntPtr CreateFile(string lpFileName, uint dwDesiredAccess, uint dwShareMode,
        IntPtr lpSecurityAttributes, uint dwCreationDisposition, uint dwFlagsAndAttributes, IntPtr hTemplateFile);

    [DllImport("kernel32.dll")]
    public static extern bool CloseHandle(IntPtr hObject);

    [StructLayout(LayoutKind.Sequential)]
    public struct HIDP_CAPS {
        public ushort Usage;
        public ushort UsagePage;
        public ushort InputReportByteLength;
        public ushort OutputReportByteLength;
        public ushort FeatureReportByteLength;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 17)]
        public ushort[] Reserved;
        public ushort NumberLinkCollectionNodes;
        public ushort NumberInputButtonCaps;
        public ushort NumberInputValueCaps;
        public ushort NumberInputDataIndices;
        public ushort NumberOutputButtonCaps;
        public ushort NumberOutputValueCaps;
        public ushort NumberOutputDataIndices;
        public ushort NumberFeatureButtonCaps;
        public ushort NumberFeatureValueCaps;
        public ushort NumberFeatureDataIndices;
    }
}
"@

# Find UPS device interface path
$devInterfaces = Get-PnpDeviceProperty -InstanceId "HID\VID_051D&PID_0002\7&38388081&0&0000" -KeyName "DEVPKEY_Device_PDOName" -ErrorAction SilentlyContinue
if (-not $devInterfaces) {
    # Try enumerating all matching interfaces
    Write-Host "Could not get PDO name, listing all properties:"
    Get-PnpDeviceProperty -InstanceId "HID\VID_051D&PID_0002\7&38388081&0&0000" | Where-Object { $_.Data -ne $null } | Format-Table KeyName, Data -AutoSize
    exit 1
}

$pdoName = $devInterfaces.Data
Write-Host "PDO Name: $pdoName"

# Build device interface path
$devPath = "\\?\HID#VID_051D&PID_0002#7&38388081&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}"
Write-Host "Device path: $devPath"

$GENERIC_READ = [uint32]0x80000000
$GENERIC_WRITE = [uint32]0x40000000
$FILE_SHARE_RW = [uint32]3
$OPEN_EXISTING = [uint32]3

$handle = [HidApi]::CreateFile($devPath, ($GENERIC_READ -bor $GENERIC_WRITE), $FILE_SHARE_RW, [IntPtr]::Zero, $OPEN_EXISTING, 0, [IntPtr]::Zero)

if ($handle -eq [IntPtr]::new(-1)) {
    $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
    Write-Host "Failed to open device (error $err). Trying read-only..."
    $handle = [HidApi]::CreateFile($devPath, $GENERIC_READ, $FILE_SHARE_RW, [IntPtr]::Zero, $OPEN_EXISTING, 0, [IntPtr]::Zero)
    if ($handle -eq [IntPtr]::new(-1)) {
        $err = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
        Write-Host "Read-only also failed (error $err)"
        exit 1
    }
}

Write-Host "Device opened successfully"

$preparsed = [IntPtr]::Zero
$ok = [HidApi]::HidD_GetPreparsedData($handle, [ref]$preparsed)
Write-Host "GetPreparsedData: $ok"

if ($ok) {
    $caps = New-Object HidApi+HIDP_CAPS
    $status = [HidApi]::HidP_GetCaps($preparsed, [ref]$caps)
    Write-Host ""
    Write-Host "=== HID Capabilities ==="
    Write-Host "UsagePage: 0x$($caps.UsagePage.ToString('X4'))"
    Write-Host "Usage: 0x$($caps.Usage.ToString('X4'))"
    Write-Host "InputReportByteLength: $($caps.InputReportByteLength)"
    Write-Host "OutputReportByteLength: $($caps.OutputReportByteLength)"
    Write-Host "FeatureReportByteLength: $($caps.FeatureReportByteLength)"
    Write-Host "NumInputValueCaps: $($caps.NumberInputValueCaps)"
    Write-Host "NumInputButtonCaps: $($caps.NumberInputButtonCaps)"
    Write-Host "NumFeatureValueCaps: $($caps.NumberFeatureValueCaps)"
    Write-Host "NumFeatureButtonCaps: $($caps.NumberFeatureButtonCaps)"

    Write-Host ""
    Write-Host "=== Feature Reports (report ID 0x00-0x3F) ==="
    $reportLen = [Math]::Max($caps.FeatureReportByteLength, 8)
    for ($id = 0; $id -le 63; $id++) {
        $buf = New-Object byte[] $reportLen
        $buf[0] = [byte]$id
        $ok = [HidApi]::HidD_GetFeature($handle, $buf, $buf.Length)
        if ($ok) {
            $hex = ($buf | ForEach-Object { $_.ToString("X2") }) -join " "
            Write-Host ("Report 0x{0:X2}: {1}" -f $id, $hex)
        }
    }

    [HidApi]::HidD_FreePreparsedData($preparsed) | Out-Null
}

[HidApi]::CloseHandle($handle) | Out-Null
