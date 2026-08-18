param(
    [string[]]$Path,
    [string]$Schema = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'knowledge/schemas/operator.schema.json')
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$script:ValidationErrors = [System.Collections.Generic.List[string]]::new()
$script:SchemaRoot = $null

function Add-ValidationError {
    param([string]$Message)
    $script:ValidationErrors.Add($Message)
}

function Get-ObjectProperty {
    param([object]$Object, [string]$Name)
    if ($null -eq $Object) {
        return $null
    }
    return $Object.PSObject.Properties[$Name]
}

function Resolve-LocalSchemaReference {
    param([string]$Reference)
    if (-not $Reference.StartsWith('#/')) {
        throw "Only local JSON Schema references are supported: $Reference"
    }

    $current = $script:SchemaRoot
    foreach ($token in $Reference.Substring(2).Split('/')) {
        $decoded = $token.Replace('~1', '/').Replace('~0', '~')
        $property = Get-ObjectProperty -Object $current -Name $decoded
        if ($null -eq $property) {
            throw "Unresolvable JSON Schema reference: $Reference"
        }
        $current = $property.Value
    }
    return $current
}

function Test-JsonType {
    param([object]$Value, [string]$Type)
    switch ($Type) {
        'null' { return $null -eq $Value }
        'boolean' { return $Value -is [bool] }
        'string' { return $Value -is [string] }
        'array' { return $Value -is [System.Array] }
        'object' {
            return $null -ne $Value -and
                -not ($Value -is [string]) -and
                -not ($Value -is [System.Array]) -and
                -not ($Value -is [System.ValueType])
        }
        'integer' {
            if ($null -eq $Value -or $Value -is [bool] -or $Value -is [string]) {
                return $false
            }
            return $Value -is [System.ValueType] -and [double]$Value -eq [math]::Floor([double]$Value)
        }
        'number' {
            return $null -ne $Value -and
                -not ($Value -is [bool]) -and
                -not ($Value -is [string]) -and
                $Value -is [System.ValueType]
        }
        default { throw "Unsupported JSON Schema type: $Type" }
    }
}

function Test-SchemaValue {
    param(
        [AllowNull()][object]$Value,
        [Parameter(Mandatory = $true)][object]$SchemaNode,
        [Parameter(Mandatory = $true)][string]$JsonPath
    )

    $referenceProperty = Get-ObjectProperty -Object $SchemaNode -Name '$ref'
    if ($null -ne $referenceProperty) {
        Test-SchemaValue -Value $Value `
            -SchemaNode (Resolve-LocalSchemaReference $referenceProperty.Value) `
            -JsonPath $JsonPath
        return
    }

    $typeProperty = Get-ObjectProperty -Object $SchemaNode -Name 'type'
    if ($null -ne $typeProperty) {
        $allowedTypes = @($typeProperty.Value)
        $typeMatches = $false
        foreach ($allowedType in $allowedTypes) {
            if (Test-JsonType -Value $Value -Type $allowedType) {
                $typeMatches = $true
                break
            }
        }
        if (-not $typeMatches) {
            Add-ValidationError "$JsonPath must have JSON type $($allowedTypes -join ' or ')."
            return
        }
    }

    $constProperty = Get-ObjectProperty -Object $SchemaNode -Name 'const'
    if ($null -ne $constProperty -and $Value -ne $constProperty.Value) {
        Add-ValidationError "$JsonPath must equal '$($constProperty.Value)'."
    }

    $enumProperty = Get-ObjectProperty -Object $SchemaNode -Name 'enum'
    if ($null -ne $enumProperty) {
        $enumMatch = $false
        foreach ($candidate in @($enumProperty.Value)) {
            if ($null -eq $candidate -and $null -eq $Value) {
                $enumMatch = $true
                break
            }
            if ($null -ne $candidate -and $null -ne $Value -and $candidate -eq $Value) {
                $enumMatch = $true
                break
            }
        }
        if (-not $enumMatch) {
            Add-ValidationError "$JsonPath is not one of the allowed values."
        }
    }

    if ($null -eq $Value) {
        return
    }

    if ($Value -is [string]) {
        $minLengthProperty = Get-ObjectProperty -Object $SchemaNode -Name 'minLength'
        if ($null -ne $minLengthProperty -and $Value.Length -lt [int]$minLengthProperty.Value) {
            Add-ValidationError "$JsonPath is shorter than $($minLengthProperty.Value) characters."
        }
        $patternProperty = Get-ObjectProperty -Object $SchemaNode -Name 'pattern'
        if ($null -ne $patternProperty -and $Value -notmatch $patternProperty.Value) {
            Add-ValidationError "$JsonPath does not match pattern '$($patternProperty.Value)'."
        }
    }

    $minimumProperty = Get-ObjectProperty -Object $SchemaNode -Name 'minimum'
    if ($null -ne $minimumProperty -and [double]$Value -lt [double]$minimumProperty.Value) {
        Add-ValidationError "$JsonPath must be at least $($minimumProperty.Value)."
    }

    if ($Value -is [System.Array]) {
        $minItemsProperty = Get-ObjectProperty -Object $SchemaNode -Name 'minItems'
        if ($null -ne $minItemsProperty -and $Value.Count -lt [int]$minItemsProperty.Value) {
            Add-ValidationError "$JsonPath must contain at least $($minItemsProperty.Value) items."
        }
        $uniqueItemsProperty = Get-ObjectProperty -Object $SchemaNode -Name 'uniqueItems'
        if ($null -ne $uniqueItemsProperty -and $uniqueItemsProperty.Value) {
            $serialized = @($Value | ForEach-Object { $_ | ConvertTo-Json -Compress -Depth 20 })
            if (@($serialized | Select-Object -Unique).Count -ne $serialized.Count) {
                Add-ValidationError "$JsonPath must contain unique items."
            }
        }
        $itemsProperty = Get-ObjectProperty -Object $SchemaNode -Name 'items'
        if ($null -ne $itemsProperty) {
            for ($index = 0; $index -lt $Value.Count; ++$index) {
                Test-SchemaValue -Value $Value[$index] -SchemaNode $itemsProperty.Value `
                    -JsonPath "$JsonPath[$index]"
            }
        }
        return
    }

    if (Test-JsonType -Value $Value -Type 'object') {
        $requiredProperty = Get-ObjectProperty -Object $SchemaNode -Name 'required'
        if ($null -ne $requiredProperty) {
            foreach ($requiredName in @($requiredProperty.Value)) {
                if ($null -eq (Get-ObjectProperty -Object $Value -Name $requiredName)) {
                    Add-ValidationError "$JsonPath.$requiredName is required."
                }
            }
        }

        $propertiesProperty = Get-ObjectProperty -Object $SchemaNode -Name 'properties'
        if ($null -ne $propertiesProperty) {
            foreach ($schemaProperty in $propertiesProperty.Value.PSObject.Properties) {
                $valueProperty = Get-ObjectProperty -Object $Value -Name $schemaProperty.Name
                if ($null -ne $valueProperty) {
                    Test-SchemaValue -Value $valueProperty.Value `
                        -SchemaNode $schemaProperty.Value `
                        -JsonPath "$JsonPath.$($schemaProperty.Name)"
                }
            }
        }

        $additionalProperty = Get-ObjectProperty -Object $SchemaNode -Name 'additionalProperties'
        if ($null -ne $additionalProperty -and $additionalProperty.Value -eq $false) {
            $allowedNames = @()
            if ($null -ne $propertiesProperty) {
                $allowedNames = @($propertiesProperty.Value.PSObject.Properties.Name)
            }
            foreach ($valueProperty in $Value.PSObject.Properties) {
                if ($valueProperty.Name -notin $allowedNames) {
                    Add-ValidationError "$JsonPath.$($valueProperty.Name) is not allowed by the schema."
                }
            }
        }
    }
}

function Get-EvidenceRecords {
    param([AllowNull()][object]$Value)
    $records = [System.Collections.Generic.List[object]]::new()

    function Visit-EvidenceNode {
        param([AllowNull()][object]$Node)
        if ($null -eq $Node -or $Node -is [string] -or $Node -is [System.ValueType]) {
            return
        }
        if ($Node -is [System.Array]) {
            foreach ($item in $Node) {
                Visit-EvidenceNode $item
            }
            return
        }

        if ($null -ne (Get-ObjectProperty $Node 'kind') -and
            $null -ne (Get-ObjectProperty $Node 'file') -and
            $null -ne (Get-ObjectProperty $Node 'match') -and
            $null -ne (Get-ObjectProperty $Node 'observation')) {
            $records.Add($Node)
            return
        }
        foreach ($property in $Node.PSObject.Properties) {
            Visit-EvidenceNode $property.Value
        }
    }

    Visit-EvidenceNode $Value
    return @($records)
}

function Test-ProvenanceEvidence {
    param([object]$Document, [string]$MetadataPath)
    function Visit-ProvenanceNode {
        param([AllowNull()][object]$Node, [string]$NodePath)
        if ($null -eq $Node -or $Node -is [string] -or $Node -is [System.ValueType]) {
            return
        }
        if ($Node -is [System.Array]) {
            for ($index = 0; $index -lt $Node.Count; ++$index) {
                Visit-ProvenanceNode $Node[$index] "$NodePath[$index]"
            }
            return
        }

        $propertiesProperty = Get-ObjectProperty $Node 'properties'
        $evidenceProperty = Get-ObjectProperty $Node 'evidence'
        if ($null -ne $propertiesProperty -and $null -ne $evidenceProperty) {
            $hasObservedValue = @(
                $propertiesProperty.Value.PSObject.Properties |
                    Where-Object { $null -ne $_.Value }
            ).Count -gt 0
            if ($hasObservedValue -and @($evidenceProperty.Value).Count -eq 0) {
                Add-ValidationError "$MetadataPath $NodePath has non-null provenance properties without evidence."
            }
        }
        foreach ($property in $Node.PSObject.Properties) {
            Visit-ProvenanceNode $property.Value "$NodePath.$($property.Name)"
        }
    }
    Visit-ProvenanceNode $Document '$'
}

function Test-PropertyBag {
    param(
        [object]$Group,
        [string[]]$AllowedNames,
        [string]$JsonPath
    )
    foreach ($stage in 'declared', 'observed', 'inferred') {
        $properties = $Group.$stage.properties
        foreach ($property in $properties.PSObject.Properties) {
            if ($property.Name -notin $AllowedNames) {
                Add-ValidationError "$JsonPath.$stage.properties.$($property.Name) is not a recognized property."
            }
            if ($null -ne $property.Value -and -not ($property.Value -is [bool])) {
                Add-ValidationError "$JsonPath.$stage.properties.$($property.Name) must be true, false, or null."
            }
        }
        foreach ($allowedName in $AllowedNames) {
            if ($null -eq (Get-ObjectProperty $properties $allowedName)) {
                Add-ValidationError "$JsonPath.$stage.properties.$allowedName is required."
            }
        }
    }
}

function Test-EvidenceFiles {
    param([object]$Document, [string]$MetadataPath)
    foreach ($evidence in Get-EvidenceRecords $Document) {
        try {
            $fullPath = Resolve-KnowledgePath -RelativePath $evidence.file
            if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
                Add-ValidationError "$MetadataPath evidence file does not exist: $($evidence.file)"
                continue
            }
            $content = [System.IO.File]::ReadAllText($fullPath)
            if ($content.IndexOf($evidence.match, [System.StringComparison]::Ordinal) -lt 0) {
                Add-ValidationError "$MetadataPath evidence match was not found in $($evidence.file): $($evidence.match)"
            }
        } catch {
            Add-ValidationError "$MetadataPath invalid evidence path '$($evidence.file)': $($_.Exception.Message)"
        }
    }
}

function Test-ArtifactPaths {
    param([object]$Implementation, [string]$MetadataPath)
    $references = @($Implementation.source.file, $Implementation.sass.artifact)
    foreach ($artifactProperty in $Implementation.measurements.artifacts.PSObject.Properties) {
        if ($null -ne $artifactProperty.Value) {
            $references += $artifactProperty.Value
        }
    }
    foreach ($reference in $references) {
        try {
            $fullPath = Resolve-KnowledgePath -RelativePath $reference
            if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
                Add-ValidationError "$MetadataPath referenced file does not exist: $reference"
            }
        } catch {
            Add-ValidationError "$MetadataPath invalid referenced path '$reference': $($_.Exception.Message)"
        }
    }
}

function Test-SassObservation {
    param([object]$Implementation, [string]$MetadataPath)
    try {
        $sassPath = Resolve-KnowledgePath -RelativePath $Implementation.sass.artifact
        if (-not (Test-Path -LiteralPath $sassPath -PathType Leaf)) {
            return
        }
        $extracted = Get-SassFeatureRecord -Sass $sassPath
        foreach ($property in $Implementation.sass.observed.features.PSObject.Properties) {
            if ($null -ne $property.Value -and $property.Value -ne $extracted.features.$($property.Name)) {
                Add-ValidationError "$MetadataPath SASS feature '$($property.Name)' disagrees with extraction."
            }
        }
        foreach ($property in $Implementation.sass.observed.counts.PSObject.Properties) {
            if ($null -ne $property.Value -and $property.Value -ne $extracted.counts.$($property.Name)) {
                Add-ValidationError "$MetadataPath SASS count '$($property.Name)' disagrees with extraction."
            }
        }
        foreach ($instruction in $Implementation.sass.observed.instructions) {
            if ($instruction -notin $extracted.instructions) {
                Add-ValidationError "$MetadataPath SASS instruction '$instruction' was not extracted."
            }
        }
        foreach ($motif in $Implementation.sass.observed.motifs) {
            $matchingMotif = @(
                $extracted.motifs | Where-Object { $_.name -eq $motif.name }
            )
            if ($matchingMotif.Count -eq 0) {
                Add-ValidationError "$MetadataPath SASS motif '$($motif.name)' was not extracted."
            }
        }
    } catch {
        Add-ValidationError "$MetadataPath SASS verification failed: $($_.Exception.Message)"
    }
}

try {
    if (-not (Test-Path -LiteralPath $Schema -PathType Leaf)) {
        throw "Schema does not exist: $Schema"
    }
    $script:SchemaRoot = Get-Content -Raw -LiteralPath $Schema | ConvertFrom-Json

    $metadataPaths = @()
    if ($null -eq $Path -or $Path.Count -eq 0) {
        $metadataPaths = @(
            Get-ChildItem -LiteralPath (Join-Path (Get-KnowledgeRepositoryRoot) 'operators') `
                -Filter 'operator.json' -File -Recurse |
                Sort-Object FullName |
                Select-Object -ExpandProperty FullName
        )
    } else {
        foreach ($candidate in $Path) {
            $metadataPaths += (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    if ($metadataPaths.Count -eq 0) {
        throw 'No operator metadata files were found.'
    }

    $hardwareProperties = @(
        'uses_cuda_core', 'uses_tensor_core', 'uses_sfu', 'uses_shared_memory',
        'uses_constant_memory', 'uses_local_memory', 'uses_warp_shuffle',
        'uses_warp_vote', 'uses_barrier', 'uses_atomic', 'uses_async_copy',
        'uses_vector_load', 'uses_vector_store'
    )
    $memoryProperties = @(
        'global_read', 'global_write', 'coalesced_read', 'coalesced_write',
        'strided_read', 'strided_write', 'gather_read', 'scatter_write',
        'shared_read', 'shared_write', 'shared_memory_tiling', 'reuses_input',
        'reuses_weights', 'intermediate_materialization'
    )
    $parallelProperties = @(
        'thread_independent', 'warp_cooperative', 'block_cooperative',
        'grid_cooperative', 'warp_reduction', 'block_reduction', 'multi_pass',
        'one_thread_per_element', 'one_warp_per_row', 'one_block_per_row',
        'tiled_execution'
    )
    $operatorNames = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )

    foreach ($metadataPath in $metadataPaths) {
        try {
            $document = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
        } catch {
            Add-ValidationError "$metadataPath is not valid JSON: $($_.Exception.Message)"
            continue
        }

        Test-SchemaValue -Value $document -SchemaNode $script:SchemaRoot -JsonPath '$'

        $operatorDirectoryName = Split-Path (Split-Path $metadataPath -Parent) -Leaf
        if ($document.identity.name -ne $operatorDirectoryName) {
            Add-ValidationError "$metadataPath identity.name must match directory '$operatorDirectoryName'."
        }
        if (-not $operatorNames.Add([string]$document.identity.name)) {
            Add-ValidationError "$metadataPath duplicates operator name '$($document.identity.name)'."
        }
        if ($document.semantics.arity -ne $document.tensor.input_count) {
            Add-ValidationError "$metadataPath semantics.arity must equal tensor.input_count."
        }
        if ('unary' -in @($document.semantics.kinds) -and $null -ne $document.algebra.commutative) {
            Add-ValidationError "$metadataPath unary commutative must be null, not false."
        }

        Test-ProvenanceEvidence -Document $document -MetadataPath $metadataPath
        Test-EvidenceFiles -Document $document -MetadataPath $metadataPath

        $implementationNames = [System.Collections.Generic.HashSet[string]]::new(
            [System.StringComparer]::OrdinalIgnoreCase
        )
        foreach ($implementation in $document.implementations) {
            $basePath = "$metadataPath implementation '$($implementation.name)'"
            if (-not $implementationNames.Add([string]$implementation.name)) {
                Add-ValidationError "$metadataPath duplicates implementation name '$($implementation.name)'."
            }
            Test-PropertyBag -Group $implementation.hardware `
                -AllowedNames $hardwareProperties -JsonPath "$basePath.hardware"
            Test-PropertyBag -Group $implementation.memory `
                -AllowedNames $memoryProperties -JsonPath "$basePath.memory"
            Test-PropertyBag -Group $implementation.parallel `
                -AllowedNames $parallelProperties -JsonPath "$basePath.parallel"
            Test-ArtifactPaths -Implementation $implementation -MetadataPath $metadataPath
            Test-SassObservation -Implementation $implementation -MetadataPath $metadataPath

            $runtimeArtifacts = @(
                $implementation.measurements.artifacts.PSObject.Properties |
                    Where-Object { $null -ne $_.Value }
            )
            $runtimeValues = @(
                $implementation.measurements.observed.properties.PSObject.Properties |
                    Where-Object { $null -ne $_.Value }
            )
            if ($runtimeArtifacts.Count -eq 0 -and $runtimeValues.Count -gt 0) {
                Add-ValidationError "$basePath has observed runtime values without runtime artifacts."
            }
        }
    }

    if ($script:ValidationErrors.Count -gt 0) {
        foreach ($validationError in $script:ValidationErrors) {
            [Console]::Error.WriteLine("ERROR: $validationError")
        }
        [Console]::Error.WriteLine("Operator metadata validation failed with $($script:ValidationErrors.Count) error(s).")
        exit 1
    }

    Write-Host "Validated $($metadataPaths.Count) operator metadata file(s) against $Schema."
    exit 0
} catch {
    [Console]::Error.WriteLine("Operator metadata validation failed: $($_.Exception.Message)")
    exit 1
}
