# Vertically balance the text blocks.
#
# After the copy rewrite the body text ended high on the slide, leaving an empty lower third, which
# reads as a layout mistake rather than deliberate whitespace. PowerPoint can measure the rendered
# height of a range, so centre each block inside the content area (below the title, above the footer)
# instead of guessing.
$ErrorActionPreference = 'Stop'
$deck = 'C:\Users\daksh\MyLife\Work\7th-sem\docs\Interim_Project_Presentation_7th-sem.pptx'

$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($deck, $false, $false, $false)

function Get-TitleShape($slide) {
    foreach ($shape in $slide.Shapes) {
        if ($shape.Type -eq 14 -and (($shape.PlaceholderFormat.Type -eq 13) -or ($shape.PlaceholderFormat.Type -eq 3))) { return $shape }
    }
    if ($null -ne $slide.Shapes.Title) { return $slide.Shapes.Title }
    return $null
}

$TOP = [single]100      # below the title
$BOTTOM = [single]500   # above the footer band
$slides = @(2, 3, 4, 8, 9, 13, 14, 18, 19, 20)

foreach ($index in $slides) {
    $slide = $pres.Slides.Item($index)
    # slide 2 keeps room under the list for the repository link
    if ($index -eq 2) { $BOTTOM = [single]470 } else { $BOTTOM = [single]500 }
    $title = Get-TitleShape $slide
    foreach ($shape in $slide.Shapes) {
        if ($null -ne $title -and $shape.Name -eq $title.Name) { continue }
        if (-not $shape.HasTextFrame) { continue }
        if (-not $shape.TextFrame.HasText) { continue }
        if ($shape.HasTable) { continue }
        # never re-centre the repository link; it belongs at the bottom of the content area
        if ($shape.TextFrame.TextRange.Text -like '*github.com*') { continue }
        $height = [single]$shape.TextFrame.TextRange.BoundHeight
        $available = $BOTTOM - $TOP
        if ($height -lt $available) {
            $shape.Top = $TOP + (($available - $height) / 2)
        } else {
            $shape.Top = $TOP
        }
        $shape.Height = $height + 6
    }
}

$pres.Save()
$pres.Close()
$app.Quit()
Write-Output 'text blocks balanced'
