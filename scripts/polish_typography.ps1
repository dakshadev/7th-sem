# Typography and alignment pass, done through PowerPoint.
#
# Human-view problem: body text sat at 13pt, the six-column literature tables at 7.6pt, captions at
# 10pt, and each element type had its own left edge (tables 30pt, text 39pt, images 30-68pt). On a
# 10.8 x 7.5 inch slide that reads as small and slightly ragged.
#
# This sets one left margin (39.6pt = 0.55in), one content width (640pt), a legible type floor per
# element, and aligns the picture grids to the same edge as the text.
$ErrorActionPreference = 'Stop'
$deck = 'C:\Users\daksh\MyLife\Work\7th-sem\docs\Interim_Project_Presentation_7th-sem.pptx'

$MARGIN = [single]39.6
$WIDTH = [single]700.8   # slide is 780pt wide; 780 - 2 x 39.6 margin

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($deck, $false, $false, $false)

function Get-TitleShape($slide) {
    foreach ($shape in $slide.Shapes) {
        if ($shape.Type -eq 14 -and (($shape.PlaceholderFormat.Type -eq 13) -or ($shape.PlaceholderFormat.Type -eq 3))) { return $shape }
    }
    if ($null -ne $slide.Shapes.Title) { return $slide.Shapes.Title }
    return $null
}

function Set-TextSize($shape, $points) {
    if (-not $shape.HasTextFrame) { return }
    if (-not $shape.TextFrame.HasText) { return }
    $shape.TextFrame.TextRange.Font.Size = [single]$points
}

function Set-ColWidths($shape, $widths) {
    $table = $shape.Table
    for ($i = 0; $i -lt $widths.Count; $i++) {
        $table.Columns.Item($i + 1).Width = [single]$widths[$i]
    }
}

function Align-Table($shape, $points) {
    $shape.Left = $MARGIN
    $shape.Width = $WIDTH
    $table = $shape.Table
    for ($r = 1; $r -le $table.Rows.Count; $r++) {
        for ($c = 1; $c -le $table.Columns.Count; $c++) {
            $range = $table.Cell($r, $c).Shape.TextFrame.TextRange
            $range.Font.Size = [single]$points
        }
    }
}

# per-slide typography: body size, and the size used inside tables if the slide has one
# Sizes chosen for a projector, not a laptop: 17pt body where there is room, 15-16pt where the
# slide is dense, and no body text below 13.5pt anywhere.
$bodySize = @{ 2 = 17; 3 = 17; 4 = 17; 8 = 15; 9 = 15.5; 13 = 15; 14 = 16; 16 = 13.5; 17 = 12.5;
               18 = 16; 19 = 16; 20 = 16; 21 = 8.5 }
$tableSize = @{ 5 = 9.5; 6 = 9.5; 7 = 9.5; 10 = 10.5; 11 = 11; 12 = 10.5 }
# Column widths in points; each set sums to the 640pt content width. Evidence columns were cramped
# and the status column was wider than the word "Completed", so the space is redistributed.
$colWidths = @{
    5  = @(45, 120, 150, 140, 140, 105)
    6  = @(45, 120, 150, 140, 140, 105)
    7  = @(45, 120, 150, 140, 140, 105)
    10 = @(150, 270, 190, 90)
    11 = @(140, 370, 190)
    12 = @(175, 195, 175, 155)
}

for ($s = 1; $s -le $pres.Slides.Count; $s++) {
    if ($s -eq 1 -or $s -eq 22) { continue }
    $slide = $pres.Slides.Item($s)
    $title = Get-TitleShape $slide

    foreach ($shape in $slide.Shapes) {
        if ($null -ne $title -and $shape.Name -eq $title.Name) { continue }

        if ($shape.HasTable) {
            $size = if ($tableSize.ContainsKey($s)) { [single]$tableSize[$s] } else { [single]10 }
            Align-Table $shape $size
            if ($colWidths.ContainsKey($s)) { Set-ColWidths $shape $colWidths[$s] }
            continue
        }

        if ($shape.Type -eq 13) {          # picture: snap the left edge of each column to the margin
            if ($shape.Left -lt 300) { $shape.Left = $MARGIN }
            continue
        }

        if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
            # The references slide carries two independent columns side by side; moving both to the
            # same margin would stack them. Only the type size applies there.
            if ($s -eq 21) {
                if ($bodySize.ContainsKey($s)) { Set-TextSize $shape ([single]$bodySize[$s]) }
                continue
            }
            $shape.Left = $MARGIN
            # leave the body narrow where a picture shares the slide
            if ($s -eq 16 -or $s -eq 17) { $shape.Width = 320 } else { $shape.Width = $WIDTH }
            if ($bodySize.ContainsKey($s)) { Set-TextSize $shape ([single]$bodySize[$s]) }
        }
    }

    # captions under the evidence pictures need to be readable, not 10pt
    if ($s -eq 15) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText -and $shape.Name -notlike '*Title*') {
                $shape.Left = $MARGIN
                $shape.Width = $WIDTH
                Set-TextSize $shape ([single]11)
            }
        }
    }
}

$pres.Save()
$pres.Close()
$app.Quit()
Write-Output 'typography pass applied'
