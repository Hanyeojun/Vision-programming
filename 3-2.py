import math as M

# 두 점의 좌표 입력
def get_num():
    x1 = float(input("x1 입력 : "))
    y1 = float(input("y1 입력 : "))
    x2 = float(input("x2 입력 : "))
    y2 = float(input("y2 입력 : "))
    return x1, y1, x2, y2

# 기울기 계산
def get_gradient(x1, y1, x2, y2):
    R = M.atan2(y2-y1, x2-x1)
    return R

# radian을 degree로 변환
def to_degree(R):
    ans = M.degrees(R)
    return ans

# main
x1, y1, x2, y2 = get_num()
rdn = get_gradient(x1, y1, x2, y2)
ans = to_degree(rdn)
print("두 점을 지나는 직선의 기울어진 각도 : ", ans, "도 입니다.")