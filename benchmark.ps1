[CmdletBinding()]
param(
    [ValidateSet('Auto', 'VS2022', 'VS2026', 'Ninja')]
    [string]$Generator = 'Auto',

    [ValidateSet('Both', 'Vulkan', 'OpenGL')]
    [string]$Backend = 'Both',

    [ValidateRange(12, 100000000)]
    [int]$Triangles = 100000,

    [uint32]$Seed = 3909885990,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$backends = if ($Backend -eq 'Both') { @('Vulkan', 'OpenGL') } else { @($Backend) }
$first = $true
foreach ($selectedBackend in $backends) {
    $arguments = @{
        Generator = $Generator
        Backend = $selectedBackend
        Triangles = $Triangles
        Seed = $Seed
        BenchmarkOnly = $true
    }
    if ($Clean -and $first) {
        $arguments.Clean = $true
    }
    & (Join-Path $PSScriptRoot 'build.ps1') @arguments
    $first = $false
}

Write-Host "Benchmark complete. CSV files are in $PSScriptRoot"
