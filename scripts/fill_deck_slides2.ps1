# Fill the two new deck slides using PowerPoint's own object model (v2).
# PowerPoint writes the package, so the file is always valid; python-pptx produced archives that
# PowerPoint refused once slides had been added to this deck.
$ErrorActionPreference = 'Stop'
$deck = 'C:\Users\daksh\MyLife\Work\7th-sem\docs\Interim_Project_Presentation_7th-sem.pptx'
$payloadPath = 'C:\Users\daksh\MyLife\Work\7th-sem\docs\assets\deck_payload.json'
# Read as UTF-8 explicitly: the default Windows PowerShell encoding mangles em dashes.
$payload = [IO.File]::ReadAllText($payloadPath, [Text.Encoding]::UTF8) | ConvertFrom-Json

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($deck, $false, $false, $false)

function Get-TitleShape($slide) {
    # Titles are ppPlaceholderTitle (13) or ppPlaceholderCenterTitle (3) depending on the layout.
    foreach ($shape in $slide.Shapes) {
        if ($shape.Type -eq 14 -and (($shape.PlaceholderFormat.Type -eq 13) -or ($shape.PlaceholderFormat.Type -eq 3))) {
            return $shape
        }
    }
    if ($null -ne $slide.Shapes.Title) { return $slide.Shapes.Title }
    return $null
}

function Set-SlideTitle($slide, $text) {
    $title = Get-TitleShape $slide
    if ($null -eq $title) { throw 'no title placeholder on slide' }
    $title.TextFrame.TextRange.Text = $text
}

function Remove-AllButTitle($slide) {
    $title = Get-TitleShape $slide
    for ($i = $slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $slide.Shapes.Item($i)
        if ($null -ne $title -and $shape.Name -eq $title.Name) { continue }
        $shape.Delete()
    }
}

function Add-Grid($slide, $rows, $left, $top, $width, $height, $fontSize) {
    $n = $rows.Count
    $c = $rows[0].Count
    $shape = $slide.Shapes.AddTable($n, $c, $left, $top, $width, $height)
    $table = $shape.Table
    for ($r = 1; $r -le $n; $r++) {
        for ($k = 1; $k -le $c; $k++) {
            $range = $table.Cell($r, $k).Shape.TextFrame.TextRange
            $range.Text = [string]$rows[$r - 1][$k - 1]
            $range.Font.Size = $fontSize
            if ($r -eq 1) { $range.Font.Bold = $true }
        }
    }
}

# ---- slide 11: methodology as a table -------------------------------------------------
$method = $pres.Slides.Item(11)
Set-SlideTitle $method $payload.methodology_title
Remove-AllButTitle $method
Add-Grid $method $payload.methodology_rows 30 95 695 320 11

# ---- new slide 12: hyperparameters ----------------------------------------------------
$hyper = $pres.Slides.Item(12).Duplicate()
$hyper.MoveTo(12)
Set-SlideTitle $hyper $payload.hyper_title
Remove-AllButTitle $hyper
Add-Grid $hyper $payload.hyper_rows 30 80 695 380 9

# ---- new slide 15: module-wise evidence ----------------------------------------------
$evidence = $pres.Slides.Item(14).Duplicate()
$evidence.MoveTo(15)
Set-SlideTitle $evidence $payload.evidence_title
Remove-AllButTitle $evidence
# Terminal captures are about 2.3:1, so 320pt wide is roughly 138pt tall: rows sit at 78 and 228,
# and the caption goes below them rather than over the second row.
# Images are 1000px wide with a 2:1 or 3.5:1 aspect, so 330pt wide is at most 164pt tall.
# Passing height -1 makes PowerPoint fall back to the image's pixel size, which squashes the
# captures. Compute the height from the real aspect ratio instead.
Add-Type -AssemblyName System.Drawing
$spots = @(@(30, 72), @(368, 72), @(30, 250), @(368, 250))
for ($i = 0; $i -lt $payload.evidence_images.Count; $i++) {
    $spot = $spots[$i]
    $bitmap = [System.Drawing.Image]::FromFile($payload.evidence_images[$i])
    $height = [int](330 * $bitmap.Height / $bitmap.Width)
    $bitmap.Dispose()
    $evidence.Shapes.AddPicture($payload.evidence_images[$i], 0, -1, $spot[0], $spot[1], 330, $height) | Out-Null
}
$caption = $evidence.Shapes.AddTextbox(1, 30, 432, 690, 66)
$caption.TextFrame.WordWrap = -1
$caption.TextFrame.TextRange.Text = ($payload.evidence_bullets -join "`r`n")
$caption.TextFrame.TextRange.Font.Size = 10

# ---- demo slide: dashboard screenshot and repository link ----------------------------
# After the two inserted slides the demonstration page sits at 17, not 16.
$demo = $pres.Slides.Item(17)
$dashboard = [System.Drawing.Image]::FromFile($payload.dashboard)
$dashHeight = [int](335 * $dashboard.Height / $dashboard.Width)
$dashboard.Dispose()
$demo.Shapes.AddPicture($payload.dashboard, 0, -1, 405, 95, 335, $dashHeight) | Out-Null
# Narrow the body text so it stops before the screenshot instead of running underneath it.
foreach ($shape in $demo.Shapes) {
    if (-not $shape.HasTextFrame) { continue }
    if (-not $shape.TextFrame.HasText) { continue }
    if ($shape.Name -like '*Title*') { continue }
    $shape.Width = 330
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.TextRange.Text = ($payload.demo_bullets -join "`r`n")
    $shape.TextFrame.TextRange.Font.Size = 11.5
}
$link = $demo.Shapes.AddTextbox(1, 30, 470, 340, 22)
$link.TextFrame.TextRange.Text = $payload.repo
$link.TextFrame.TextRange.Font.Size = 10
$link.TextFrame.TextRange.ActionSettings.Item(1).Hyperlink.Address = $payload.repo

$pres.Save()
$count = $pres.Slides.Count
$pres.Close()
$app.Quit()
Write-Output "saved with $count slides"
