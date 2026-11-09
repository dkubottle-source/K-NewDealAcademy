"""Portable Korean inspection-history PDF export."""
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import json
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def history_pdf(frame):
    font = 'InspectionKorean'
    if font not in pdfmetrics.getRegisteredFontNames():
        path = Path('C:/Windows/Fonts/malgun.ttf')
        if path.exists():
            pdfmetrics.registerFont(TTFont(font, str(path)))
        else:
            font = 'HYSMyeongJo-Medium'
            if font not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(UnicodeCIDFont(font))
    normal = ParagraphStyle('body', fontName=font, fontSize=9, leading=14, wordWrap='CJK')
    title = ParagraphStyle('title', parent=normal, fontSize=18, leading=26, spaceAfter=12)
    def paragraph(value):
        return Paragraph(escape(str(value)), normal)
    heading = ParagraphStyle('event', parent=normal, keepWithNext=True)
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36,
                                 topMargin=40, bottomMargin=40)
    content = [Paragraph('냉간압연 AI 점검 이력 보고서', title),
        paragraph(f'전체 기록 {len(frame):,}건 중 최신 {min(200, len(frame)):,}건'),
        paragraph('공개 시뮬레이션 데이터 기반 점검 참고 자료. 기록 시각은 프로그램 실행 시각(UTC)이며 측정 시각은 제공되지 않습니다.'),
        Spacer(1, 14)]
    for _, row in frame.head(200).iterrows():
        content.append(Paragraph(escape(f"사례 {row.sample_index} / {row.prediction} / AI 점수 {row.score:.2%} / 이진 임계값 {row.threshold:.6g}"), heading))
        rows = [[paragraph('기록 시각 UTC'), paragraph(row.recorded_at)],
                [paragraph('모델 / 출처'), paragraph(f'{row.model} / {row.source}')],
                [paragraph('실제 라벨 / 단계'), paragraph(f'{row.actual_label} / {row.step}')],
                [paragraph('점검 참고 변수'), paragraph(row.inspection)]]
        for key, label in [('source_sha256', '파일 SHA-256'), ('source_row_id', '원본 행 식별자'),
                           ('model_version', '실제 모델 버전'), ('inspection_status', '점검 상태'),
                           ('operator_result', '작업자 확인 결과'), ('inspection_note', '점검 메모'),
                           ('inspection_updated_at', '점검 수정 시각 UTC')]:
            rows.append([paragraph(label), paragraph(row.get(key, '미확인'))])
        for key,label in [('risk_level','연구용 위험도'),('risk_caution_threshold','주의 점수 임계값'),
                          ('risk_warning_threshold','경고 점수 임계값'),('risk_policy_version','위험도 정책 버전'),
                          ('model_features','모델 특징 변수')]:
            value=row.get(key,'미확인')
            if key=='model_features' and value!='미확인':
                features=json.loads(value)
                value=f'{len(features)}개: '+', '.join(features)
            rows.append([paragraph(label),paragraph(value)])
        table = Table(rows, colWidths=[100, 423], splitInRow=1)
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
            ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#edf3f5')),
            ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#ccd6da')),
            ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        content.extend([table, Spacer(1, 14)])
    content.append(paragraph('변수 순위는 정상 학습 분포와의 편차를 나타냅니다. 원인과 고장 위치를 확정할 근거는 별도 점검으로 확인해야 합니다.'))
    def footer(canvas, doc):
        canvas.setFont(font, 8)
        canvas.drawRightString(A4[0]-36, 22, f'{doc.page}')
    document.build(content, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
