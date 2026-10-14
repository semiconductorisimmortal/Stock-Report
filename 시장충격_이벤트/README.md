# 시장 충격 이벤트 아카이브
중대한 호재·악재가 실제 시장에 미친 영향을 한국어 Markdown으로 축적합니다. GitHub에서 문서를 클릭하면 다운로드 없이 읽을 수 있습니다.

[저장소 홈](../README.md) · [미국 증시 일일 보고서](../premarket/) · [AI·로보틱스 주간 보고서](../physical-ai/)

## 사건 목록
날짜는 대표 시장 반응일(KST)이며 발표일·작성일은 개별 문서에 별도로 표시합니다. 등락률은 정규장 전 거래일 대비입니다.

| 날짜 | 사건명 | 관련 기업 | 시장 반응 | 문서 |
| --- | --- | --- | --- | --- |
<!-- MARKET_IMPACT_ROWS_START: update only verified event rows, avoid duplicate event_id -->
| 2026-10-13 | CXMT 4F² 서버 DRAM 계획과 한국 메모리주 급락 | CXMT·삼성전자·SK하이닉스·Micron | 삼성전자 −7.39%, SK하이닉스 −10.09%, 코스피 −5.46%; 복합 요인 | [심층 보고서](2026-10-13_CXMT_DRAM.md) |
<!-- MARKET_IMPACT_ROWS_END -->

## 선정·갱신 원칙
- 중대한 산업·실적·정책·기술·거시 사건의 시장 파급력과 예상 대비 변화로 선정합니다. 변동률만으로 생성하지 않습니다.
- 호재·악재를 모두 포함하고 기업·공급자·구매자 관점을 구별합니다.
- 사건이나 의미 있는 변화가 없으면 파일·알림·인덱스 행을 생성하지 않습니다.
- 새 파일명은 `YYYY-MM-DD_사건키워드.md`. 대표 반응일, 최초 발표·보도 시각, 작성·검증일을 구분합니다.
- 사건별 안정적인 `event_id`를 문서 주석에 기록합니다. 신규 작성 전 기존 파일과 사건 ID·기업·발표 내용을 확인합니다. 같은 사건의 다음 날 주가 변동만으로 별도 파일을 만들지 않습니다.
- 실제 출시·고객 인증·실적 변경·정책 확정 등 의미 있는 새 증거가 있으면 같은 파일의 날짜별 후속 평가와 변경 이력에 추가합니다. 인덱스는 사건당 한 행을 유지합니다.
- 사전 전망은 고정된 원본 커밋 링크와 당시 실제 발행 시각을 보존합니다. 사후 결과로 최초 전망을 소급 수정하지 않습니다. 과거 시점 재구성은 실시간 예측 성과로 평가하지 않습니다.
- 기술적 사실, 회사 주장, 언론 전달, 분석·전망, 미확인을 구별합니다. 주가 동행만으로 단일 원인을 확정하지 않습니다.
- 시장 수치는 거래일·종목·시장·정규장/장중·시간외·통화·관측 시각·자료 출처를 명시합니다.

## 향후 예약 기록 절차
대상은 `semiconductorisimmortal/Stock-Report`의 `main`과 이 폴더입니다. 사용자가 지정한 기존 예약은 월~금 22:30 KST입니다. 이 README는 예약 생성·변경 또는 실행 성공의 증거가 아닙니다.

1. 실행 환경에서 GitHub 읽기·쓰기 도구와 대상 저장소 권한을 실제 확인합니다.
2. 최신 인덱스·기존 사건·원본 전망을 읽고 중복 여부를 확인합니다.
3. 새 사건 또는 실질적 후속 정보가 있을 때만 출처 검증 후 작성합니다.
4. 최신 브랜치와 파일 SHA를 읽어 충돌을 검사합니다. 다른 예약의 변경을 덮어쓰지 않습니다. 가능하면 사건 파일과 인덱스를 하나의 커밋에 반영합니다.
5. 커밋 후 브랜치의 커밋 SHA와 파일 내용·인덱스 링크를 재조회합니다. 실패하거나 재검증하지 못한 경우 저장 성공이라고 알리지 않습니다.
6. 알림에는 검증된 GitHub 문서 URL·커밋 SHA와 저장 상태를 기록합니다. Google Drive에는 저장하지 않습니다.
7. 최상위 README를 갱신할 필요가 있으면 이 아카이브 링크에 한정하며 기존 미국 증시·Physical AI 발행 블록과 자동화 표식을 보존합니다.

## 자동화 검증 상태
**2026-10-14:** 현재 대화의 수동 GitHub 쓰기 검증과 기존 예약의 클라우드 자동 쓰기는 별개입니다. [공식 예약 문서](https://learn.chatgpt.com/docs/automations)는 웹 예약에서 해당 대화에 제공된 연결 도구·플러그인을 사용할 수 있다고 설명하지만, 개별 예약의 GitHub 쓰기 권한이나 성공을 보증하지 않습니다.

해당 예약의 실제 실행 로그에서 GitHub 쓰기 호출, 성공 응답, 커밋 SHA와 저장소 재조회가 확인될 때만 **자동 저장 검증 완료**로 변경합니다. 현재 그 증거는 확보되지 않았습니다. 이 작업에서는 기존 예약을 중복 생성하지 않습니다.

<!-- MARKET_IMPACT_AUTOMATION_CONTRACT: GitHub only. No event/no meaningful update means no write. Deduplicate by stable event_id and existing files. Preserve initial forecasts; append dated follow-ups. SHA-aware writes, preserve unrelated files, verify branch commit and rendered links before claiming success. Manual write success does not prove scheduled cloud write success. -->
