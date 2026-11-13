# Final pass over the deck, done through PowerPoint so the package stays valid.
#   * outline slide states what each section holds, with the repository link
#   * a bullet character on every content paragraph (skip titles, tables, references, thanks)
#   * an honest note about the demo recording on the demonstration slide
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

# ---- outline slide: contents, with the repository link --------------------------------
function Remove-RepoLinks($slide) {
    # Re-running this script must not stack a second copy of the repository link on the slide.
    for ($i = $slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $slide.Shapes.Item($i)
        if (-not $shape.HasTextFrame) { continue }
        if (-not $shape.TextFrame.HasText) { continue }
        if ($shape.TextFrame.TextRange.Text -like '*github.com/dakshadev/7th-sem*') { $shape.Delete() }
    }
}

Remove-RepoLinks $pres.Slides.Item(2)
$outline = $pres.Slides.Item(2)
$body = $null
foreach ($shape in $outline.Shapes) {
    if ($shape.HasTextFrame -and $shape.TextFrame.HasText -and $shape.Name -notlike '*Title*') { $body = $shape }
}
if ($null -eq $body) { throw 'outline body not found' }
$body.TextFrame.TextRange.Text = ($payload.outline -join "`r`n")
$body.TextFrame.TextRange.Font.Size = 13
# Sit the link below the list, not inside it: the balanced list can run to ~470pt.
$link = $outline.Shapes.AddTextbox(1, 39.6, 496, 380, 20)
$link.TextFrame.TextRange.Text = 'Repository: ' + $payload.repo
$link.TextFrame.TextRange.Font.Size = 10
$link.TextFrame.TextRange.ActionSettings.Item(1).Hyperlink.Address = $payload.repo

# ---- bullets everywhere they make sense ---------------------------------------------
$skipSlides = @(1, 21, 22)   # title, references, thank-you
for ($s = 1; $s -le $pres.Slides.Count; $s++) {
    if ($skipSlides -contains $s) { continue }
    $slide = $pres.Slides.Item($s)
    $title = Get-TitleShape $slide
    foreach ($shape in $slide.Shapes) {
        if (-not $shape.HasTextFrame) { continue }
        if (-not $shape.TextFrame.HasText) { continue }
        if ($null -ne $title -and $shape.Name -eq $title.Name) { continue }
        if ($shape.HasTable) { continue }
        $range = $shape.TextFrame.TextRange
        # Paragraphs() needs explicit arguments through the COM wrapper.
        $paragraphs = $range.Paragraphs(-1, -1)
        foreach ($paragraph in $paragraphs) {
            $text = $paragraph.Text
            if ([string]::IsNullOrWhiteSpace($text)) { continue }
            # numbered objectives keep their numerals, table-of-contents rows keep their dashes
            if ($text -match '^\s*\d+\.') { continue }
            $paragraph.ParagraphFormat.Bullet.Visible = -1
            $paragraph.ParagraphFormat.Bullet.Character = 8226
            $paragraph.ParagraphFormat.Bullet.RelativeSize = 0.9
        }
    }
}

$pres.Save()
$count = $pres.Slides.Count
$pres.Close()
$app.Quit()
Write-Output "saved with $count slides"
