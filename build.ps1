[CmdletBinding()]
param(
    [ValidateSet('Auto', 'VS2022', 'VS2026', 'Ninja')]
    [string]$Generator = 'Auto',

    [ValidateSet('Vulkan', 'OpenGL')]
    [string]$Backend = 'Vulkan',

    [ValidateRange(12, 100000000)]
    [int]$Triangles = 100000,

    [uint32]$Seed = 3909885990,

    [switch]$Benchmark,
    [switch]$BenchmarkOnly,
    [switch]$NoRun,
    [switch]$Clean,

    [string]$EpochGuiSource = '',
    [switch]$NoEpochGui
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-VcpkgRoot {
    if ($env:VCPKG_ROOT) {
        $configured = [System.IO.Path]::GetFullPath($env:VCPKG_ROOT)
        if (Test-Path (Join-Path $configured 'scripts/buildsystems/vcpkg.cmake')) {
            return $configured
        }
    }

    $candidates = @(
        'C:\Users\iammi\source\repos\vcpkg',
        (Join-Path $PSScriptRoot 'vcpkg'),
        (Join-Path (Split-Path $PSScriptRoot -Parent) 'vcpkg'),
        (Join-Path $HOME 'source\repos\vcpkg'),
        (Join-Path $HOME 'vcpkg'),
        'C:\vcpkg'
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path (Join-Path $candidate 'scripts/buildsystems/vcpkg.cmake'))) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }

    throw @'
Unable to locate vcpkg.
Set VCPKG_ROOT to an existing vcpkg checkout, for example:
  $env:VCPKG_ROOT = 'C:\Users\iammi\source\repos\vcpkg'
CMake will run manifest installation automatically during configure.
'@
}

function Get-CMakeGeneratorHelp {
    $output = & cmake --help 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to query CMake generators. Ensure cmake is available on PATH.'
    }
    return ($output | Out-String)
}

function Resolve-GeneratorSelection {
    param(
        [Parameter(Mandatory)]
        [string]$Requested
    )

    $help = Get-CMakeGeneratorHelp
    $hasVS2026 = $help.Contains('Visual Studio 18 2026')
    $hasVS2022 = $help.Contains('Visual Studio 17 2022')
    $hasNinja = $help -match '(?m)^\s*\*?\s*Ninja\s*='

    switch ($Requested) {
        'Auto' {
            if ($hasVS2026) { return 'VS2026' }
            if ($hasVS2022) { return 'VS2022' }
            if ($hasNinja) { return 'Ninja' }
            throw 'No supported Visual Studio or Ninja CMake generator was found.'
        }
        'VS2026' {
            if (-not $hasVS2026) {
                throw 'This CMake installation does not provide the Visual Studio 18 2026 generator. Use -Generator VS2022 or update CMake.'
            }
            return 'VS2026'
        }
        'VS2022' {
            if (-not $hasVS2022) {
                throw 'This CMake installation does not provide the Visual Studio 17 2022 generator.'
            }
            return 'VS2022'
        }
        'Ninja' {
            if (-not $hasNinja) {
                throw 'This CMake installation does not provide the Ninja generator.'
            }
            return 'Ninja'
        }
        default {
            throw "Unsupported generator selection: $Requested"
        }
    }
}

function Get-BuildConfiguration {
    param(
        [Parameter(Mandatory)]
        [string]$SelectedGenerator
    )

    switch ($SelectedGenerator) {
        'VS2026' {
            return @{
                Preset = 'windows-vs2026-release'
                Directory = 'build/windows-vs2026'
                DirectoryPrefix = 'build/windows-vs2026/Release'
            }
        }
        'VS2022' {
            return @{
                Preset = 'windows-vs2022-release'
                Directory = 'build/windows-vs2022'
                DirectoryPrefix = 'build/windows-vs2022/Release'
            }
        }
        'Ninja' {
            return @{
                Preset = 'windows-ninja-release'
                Directory = 'build/windows-ninja-release'
                DirectoryPrefix = 'build/windows-ninja-release'
            }
        }
        default {
            throw "Unsupported generator selection: $SelectedGenerator"
        }
    }
}

Push-Location $PSScriptRoot
try {
    $env:VCPKG_ROOT = Resolve-VcpkgRoot
    $selectedGenerator = Resolve-GeneratorSelection -Requested $Generator
    $configuration = Get-BuildConfiguration -SelectedGenerator $selectedGenerator

    $preset = $configuration.Preset
    $buildDirectory = Join-Path $PSScriptRoot $configuration.Directory
    $targetName = if ($Backend -eq 'OpenGL') { 'epoch_voxel_opengl_demo' } else { 'epoch_voxel_vulkan_demo' }
    $executableName = "$targetName.exe"
    $executable = Join-Path $PSScriptRoot (Join-Path $configuration.DirectoryPrefix $executableName)

    if ($Clean) {
        Remove-Item $buildDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "VCPKG_ROOT=$env:VCPKG_ROOT"
    Write-Host "Generator=$selectedGenerator"
    Write-Host "Backend=$Backend"
    Write-Host "Configuring with preset: $preset"

    $configureArguments = @('--preset', $preset)
    if ($NoEpochGui) {
        $configureArguments += '-DEPOCH_VISUALIZER_USE_EPOCHGUI=OFF'
    }
    elseif ($EpochGuiSource) {
        $resolvedEpochGui = [System.IO.Path]::GetFullPath($EpochGuiSource)
        $configureArguments += "-DEPOCHGUI_SOURCE_DIR=$resolvedEpochGui"
    }

    & cmake @configureArguments
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE."
    }

    cmake --build --preset $preset --target $targetName
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE."
    }

    if (-not $NoRun) {
        if (-not (Test-Path $executable)) {
            throw "Build succeeded, but the executable was not found: $executable"
        }

        $applicationArguments = @('--triangles', $Triangles.ToString(), '--seed', $Seed.ToString())
        if ($BenchmarkOnly) {
            $applicationArguments += '--benchmark-only'
        }
        elseif ($Benchmark) {
            $applicationArguments += '--benchmark'
        }
        & $executable @applicationArguments
        if ($LASTEXITCODE -ne 0) {
            throw "$executableName exited with code $LASTEXITCODE."
        }
    }
}
finally {
    Pop-Location
}
