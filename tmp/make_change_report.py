from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

out = Path('F:/sih26174/output/pdf')
out.mkdir(parents=True, exist_ok=True)
pdf = out / 'SIH26174_change_summary.pdf'

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=20, leading=24, textColor=colors.HexColor('#123B5D')))
styles.add(ParagraphStyle(name='Sub', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10, textColor=colors.HexColor('#52606D')))
styles.add(ParagraphStyle(name='H', parent=styles['Heading2'], fontSize=13, leading=16, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor('#123B5D')))
styles.add(ParagraphStyle(name='Body2', parent=styles['BodyText'], fontSize=9.5, leading=13, spaceAfter=5))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=8.5, leading=11))

def P(text, style='Body2'):
    return Paragraph(text, styles[style])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D9E2EC'))
    canvas.line(18*mm, 14*mm, 192*mm, 14*mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#52606D'))
    canvas.drawString(18*mm, 9*mm, 'SIH 26174 - Change Summary')
    canvas.drawRightString(192*mm, 9*mm, f'Page {doc.page}')
    canvas.restoreState()

story = []
story += [P('SIH 26174 - Implementation Change Summary', 'TitleCenter'), Spacer(1, 3*mm), P('AI Human Activity Recognition for On-board BAS Experiments', 'Sub'), P('Prepared for merge review | 26 September 2026', 'Sub'), Spacer(1, 9*mm)]
story += [P('Purpose', 'H'), P('This document records the repository setup, model installation, code changes, experiment-flow changes, and current limitations from the beginning of the work until now. It is intended to be shared with the person merging the changes.'), P('<b>Important:</b> The current Git working tree is clean. The listed functionality is present in the current repository state; there are no pending uncommitted changes at the time of this report.'), Spacer(1, 3*mm)]

story += [P('1. Initial setup and execution', 'H')]
setup = [
    ['Item', 'Action / result'],
    ['Python environment', 'Created/used a project virtual environment named venv. A broken environment was identified and the recommended fix was to recreate it and reinstall requirements.'],
    ['Dependencies', 'Installed from requirements.txt. The project uses OpenCV, NumPy, MediaPipe, Pydantic, PyYAML, pytest, and related packages.'],
    ['MediaPipe model', 'Added models/mediapipe/hand_landmarker.task so hand detection can run offline.'],
    ['Demo entry point', 'scripts/run_demo.py is the main live demo. It supports webcam input, file input, mute/headless modes, recording, and streaming options.'],
]
t = Table([[P(c, 'Small') for c in row] for row in setup], colWidths=[42*mm, 132*mm], repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D9EAF7')),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#B8C7D1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t]

story += [P('2. File-wise code changes', 'H')]
changes = [
    ['File', 'Changes made'],
    ['configs/hsv_ranges.yaml', 'Added/tuned red_box and yellow_box HSV ranges. Added main_box detection for a bright blue/cyan main-box marker, with minimum area and frame-area limits.'],
    ['src/har_space/perception/color_detector.py', 'Added configurable minimum contour area, minimum/maximum frame-area ratio, and rectangularity filtering to reject irregular blobs such as faces or clothing.'],
    ['src/har_space/events/interactions.py', 'Changed hold-duration measurement to time.monotonic(). This fixes webcam timing, where CAP_PROP_POS_MSEC can be zero or unreliable.'],
    ['src/har_space/engine/tracker.py', 'Added mapping for object predicates and changed evaluation to the current FSM state only. Repeated main-box detections no longer re-trigger completed states.'],
    ['scripts/run_demo.py', 'Imported Event; publishes main-box-present events; tracks red/yellow positions; detects removal outside the main box; detects stable left/right placement; requires hand contact for removal; uses main-box-relative left/right.'],
    ['configs/experiment_sample.yaml', 'Removed opening-main-box and hold-red-box behavior. Added the intended main-box, yellow removal/left placement, and red removal/right placement sequence.'],
    ['configs/experiment_demo.yaml', 'Updated demo configuration to the same five-step experiment sequence.'],
]
t = Table([[P(c, 'Small') for c in row] for row in changes], colWidths=[57*mm, 117*mm], repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D9EAF7')),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#B8C7D1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t]

story += [PageBreak(), P('3. Current experiment FSM', 'H'), P('The current intended sequence is:'), Spacer(1, 2*mm)]
fsm = [['State', 'Condition / action', 'Next state'], ['MAIN_BOX_PRESENT', 'Bright blue/cyan main box is detected.', 'TAKE_OUT_YELLOW'], ['TAKE_OUT_YELLOW', 'Yellow box first appears outside the main box while hand contact is present.', 'PLACE_YELLOW_LEFT'], ['PLACE_YELLOW_LEFT', 'Yellow box is left of the main-box boundary and stable for about 10 frames.', 'TAKE_OUT_RED'], ['TAKE_OUT_RED', 'Red box first appears outside the main box while hand contact is present.', 'PLACE_RED_RIGHT'], ['PLACE_RED_RIGHT', 'Red box is right of the main-box boundary and stable for about 10 frames.', 'COMPLETE'], ['COMPLETE', 'All required states are completed.', 'End']]
t = Table([[P(c, 'Small') for c in row] for row in fsm], colWidths=[43*mm, 83*mm, 48*mm], repeatRows=1)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#D9EAF7')),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#B8C7D1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
story += [t, Spacer(1, 5*mm), P('4. How the camera demo is expected to work', 'H'), P('1. Place a cardboard main box in view with a thick bright blue/cyan border or tape. The red and yellow boxes may remain hidden inside initially.<br/>2. Start the demo with <font name="Courier">python scripts/run_demo.py</font>.<br/>3. Once the main box is detected, Step 1 completes automatically; no opening action is required.<br/>4. Pick up the yellow box while keeping enough of it visible to the camera. Move it outside the main-box boundary and place it on the left.<br/>5. Keep the yellow box stable for about 10 frames. Then repeat the process with the red box and place it on the right.'), P('5. Current limitations and merge notes', 'H'), P('- The system is still heuristic/OpenCV-based; it is not yet a trained object-detection model.<br/>- The main-box detector assumes a visible bright blue/cyan marker and sufficient contrast with the background.<br/>- The red/yellow detector can still fail under poor lighting, occlusion, or strong color confusion.<br/>- Removal and placement are inferred from bounding-box positions, hand contact, and short-term stability; they are not yet learned actions.<br/>- The code should be tested with the actual physical box, camera angle, lighting, and background before a final merge.<br/>- The MediaPipe warnings printed at startup are warnings; the previous runtime crash was fixed by importing Event.'), P('6. Recommended merge verification', 'H'), P('Run the demo, verify the blue/cyan main box is detected, confirm that Step 1 advances without showing the contents, then test yellow-left and red-right placement. Also inspect the generated runs/<timestamp>/steps.jsonl and summary.json files.'), Spacer(1, 4*mm), P('This report describes the current repository state and the intended behavior of the implemented changes.', 'Small')]

doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=19*mm, title='SIH 26174 Change Summary', author='Codex')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(pdf)
