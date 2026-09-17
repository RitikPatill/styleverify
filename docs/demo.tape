# StyleVerify demo tape
# Regenerate with: vhs docs/demo.tape  (requires https://github.com/charmbracelet/vhs)
# Note: run `PYTHONPATH=src uvicorn main:app` in a separate terminal before the curl demo.
# The fixture image at tests/fixtures/face_a.jpg is a 64x64 synthetic PNG.

Output docs/demo.gif
Set FontSize 14
Set Width 1000
Set Height 600

Type "# StyleVerify — face verification that survives artistic filters"
Enter
Sleep 1s

Type "# CLI: compare two images"
Enter
Type "PYTHONPATH=src python verify.py tests/fixtures/face_a.jpg tests/fixtures/face_a.jpg"
Enter
Sleep 3s

Type "# Style demo grid — synthesises original + 5 styled variants"
Enter
Type "PYTHONPATH=src python scripts/demo_styles.py --output /tmp/demo_grid.png"
Enter
Sleep 4s

Type "# REST API — start the server first: PYTHONPATH=src uvicorn main:app"
Enter
Type "curl -s -X POST http://localhost:8000/verify \\"
Enter
Type "  -F 'image1=@tests/fixtures/face_a.jpg' \\"
Enter
Type "  -F 'image2=@tests/fixtures/face_a.jpg' | python -m json.tool"
Enter
Sleep 2s
