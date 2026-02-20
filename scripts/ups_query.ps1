# Query UPS status via multiple Windows APIs
Write-Host "=== Win32_Battery (WMI) ==="
$battery = Get-WmiObject -Namespace root/cimv2 -Class Win32_Battery
if ($battery) {
    Write-Host "Name: $($battery.Name)"
    Write-Host "DeviceID: $($battery.DeviceID)"
    Write-Host "Charge: $($battery.EstimatedChargeRemaining)%"
    Write-Host "Runtime: $($battery.EstimatedRunTime) min"
    Write-Host "BatteryStatus: $($battery.BatteryStatus)"
    Write-Host "TimeOnBattery: $($battery.TimeOnBattery)"
    Write-Host "DesignVoltage: $($battery.DesignVoltage) mV"
    Write-Host "Chemistry: $($battery.Chemistry)"
    Write-Host "Availability: $($battery.Availability)"
    Write-Host "Status: $($battery.Status)"
} else {
    Write-Host "No Win32_Battery found"
}

Write-Host ""
Write-Host "=== SystemInformation.PowerStatus (.NET) ==="
Add-Type -AssemblyName System.Windows.Forms
$ps = [System.Windows.Forms.SystemInformation]::PowerStatus
Write-Host "PowerLineStatus: $($ps.PowerLineStatus)"
Write-Host "ChargeStatus: $($ps.BatteryChargeStatus)"
Write-Host "LifePercent: $($ps.BatteryLifePercent)"
Write-Host "LifeRemaining: $($ps.BatteryLifeRemaining) sec"
Write-Host "FullLifetime: $($ps.BatteryFullLifetime) sec"

Write-Host ""
Write-Host "=== PnP Devices (Battery class) ==="
Get-PnpDevice | Where-Object { $_.Class -eq "Battery" } | Format-Table Status, FriendlyName, InstanceId -AutoSize

Write-Host ""
Write-Host "=== HID UPS devices ==="
Get-PnpDevice | Where-Object { $_.FriendlyName -match "UPS|APC" } | Format-Table Status, Class, FriendlyName, InstanceId -AutoSize
