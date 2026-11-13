# Replace the wordy body text with presentation-register copy, then size it for a projector.
# Full sentences read well in the paper; on a slide they force the type down to a size nobody at the
# back can read. These keep the claim, drop the clauses.
$ErrorActionPreference = 'Stop'
$deck = 'C:\Users\daksh\MyLife\Work\7th-sem\docs\Interim_Project_Presentation_7th-sem.pptx'
$payload = [IO.File]::ReadAllText('C:\Users\daksh\MyLife\Work\7th-sem\docs\assets\deck_payload.json', [Text.Encoding]::UTF8) | ConvertFrom-Json

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($deck, $false, $false, $false)

function Get-TitleShape($slide) {
    foreach ($shape in $slide.Shapes) {
        if ($shape.Type -eq 14 -and (($shape.PlaceholderFormat.Type -eq 13) -or ($shape.PlaceholderFormat.Type -eq 3))) { return $shape }
    }
    if ($null -ne $slide.Shapes.Title) { return $slide.Shapes.Title }
    return $null
}

function Set-Body($slide, $lines, $size) {
    $title = Get-TitleShape $slide
    foreach ($shape in $slide.Shapes) {
        if ($null -ne $title -and $shape.Name -eq $title.Name) { continue }
        if (-not $shape.HasTextFrame) { continue }
        if (-not $shape.TextFrame.HasText) { continue }
        if ($shape.HasTable) { continue }
        $shape.TextFrame.TextRange.Text = ($lines -join "`r`n")
        $shape.TextFrame.TextRange.Font.Size = [single]$size
        return
    }
}

Set-Body $pres.Slides.Item(3) $payload.slide_text.s3 20
Set-Body $pres.Slides.Item(4) $payload.slide_text.s4 20
Set-Body $pres.Slides.Item(8) $payload.slide_text.s8 15.5
Set-Body $pres.Slides.Item(14) $payload.slide_text.s14 18
Set-Body $pres.Slides.Item(18) $payload.slide_text.s18 17
Set-Body $pres.Slides.Item(19) $payload.slide_text.s19 18
Set-Body $pres.Slides.Item(20) $payload.slide_text.s20 18

$pres.Save()
$pres.Close()
$app.Quit()
Write-Output 'slide copy rewritten'
