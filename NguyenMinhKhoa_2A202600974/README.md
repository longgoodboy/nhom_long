# Bai Ca Nhan

Thu muc nay chua toan bo bai ca nhan, tach rieng khoi `src/` cua bai nhom.

## Cau truc

- `tasks/`: code cho Task 1-10 cua bai ca nhan
- `checklist.md`: checklist thuc hien

## Ghi chu

- Du lieu Task 1 nam o `data/landing/legal/`
- Du lieu Task 2 nam o `data/landing/news/` va dung URL bai bao that
- `src/` duoc giu cac file cau noi nhe de tranh vo test cu va de bai nhom co the goi lai neu can

## Toi uu de lay diem cao

- Pipeline da toi uu cho truy van co dau va khong dau
- Co mo rong query bang tu dong nghia de bat duoc nhieu cach hoi khac nhau
- Retrieval uu tien dung nhom nguon: cau hoi phap ly uu tien `legal`, cau hoi nghe si uu tien `news`
- Hybrid retrieval gom semantic + BM25 + reranking + fallback
- Generation tra loi kem citation tu nguon da truy xuat
