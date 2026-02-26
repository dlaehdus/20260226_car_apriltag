
<img width="792" height="879" alt="스크린샷 2026-02-26 14-05-02" src="https://github.com/user-attachments/assets/5709dd58-480c-4e65-b026-a6d580edcdf3" />
위 사진 apriltag.py은 에어프릴 태그를 시각화한것으로 좌표축과 해당 에어프릴태그를 정육각형으로 시각화한 코드

<img width="917" height="684" alt="스크린샷 2026-02-26 14-06-57" src="https://github.com/user-attachments/assets/a4dd2139-d231-400f-8b93-5c531767de5d" />
위 사진은 car_apriltag_1번 코드로 에어프릴태그를 차량 번호판 기준으로 하려했는데 객체인식이 잘안됐음 따라서 객체인식을 잘하도록 학습을 시켜야했음

<img width="962" height="876" alt="스크린샷 2026-02-26 14-08-33" src="https://github.com/user-attachments/assets/f63f7297-f555-4d7f-bc31-4ee800e99bd8" />
위 사진은 car_apriltag_2번 코드로 객체의 인식률은 높혔지만 아직도 정밀한 방향표시가 되지않은 문제가 발생함따라서 번호판의 글자들에 사각박스를 형성하고
해당 박스의 모서리 기준으로 방향을 잡도록 하기로함



<img width="1222" height="711" alt="스크린샷 2026-02-26 14-00-39" src="https://github.com/user-attachments/assets/79ca35f2-4fd5-4425-9ebf-e86f2adf06b3" />
위사진은 car_apriltag_4번 코드로 쓸모없는 값도 인식함

<img width="1221" height="691" alt="스크린샷 2026-02-26 14-00-20" src="https://github.com/user-attachments/assets/e8f16173-3aff-43a9-88f5-1581789bc361" />
위사진은 car_apriltag_5번 코드로 차량번호판의 형식처럼 연속적인 패턴이 아니면 인식하지 않는 필터를 적용해 번호판만 인식
