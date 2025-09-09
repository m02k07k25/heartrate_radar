import numpy as np

# 확인하고 싶은 .npy 파일의 경로를 지정합니다.
# 경로에 한글이 포함되어 있으므로, 문자열 앞에 r을 붙여주는 것이 좋습니다.
file_path = 'record3/train/data/1.npy'


def _preview_ndarray(arr: np.ndarray, max_items: int = 10) -> str:
    """NumPy 배열 미리보기 문자열을 생성합니다."""
    try:
        size = arr.size
        if size <= max_items:
            return np.array2string(arr, threshold=max_items, edgeitems=max_items, suppress_small=True)
        flat = arr.ravel()
        head = flat[:max_items]
        return f"앞 {max_items}개: " + np.array2string(head, threshold=max_items, suppress_small=True)
    except Exception as e:
        return f"미리보기 생성 중 오류: {e}"


def _safe_repr(value, max_len: int = 300) -> str:
    s = repr(value)
    return s if len(s) <= max_len else s[:max_len] + '...'


try:
    # allow_pickle=True 옵션을 추가하여 객체 배열을 로드할 수 있도록 허용합니다.
    data = np.load(file_path, allow_pickle=True)

    # 기본 정보 출력
    print(f"파일 경로: {file_path}")
    if isinstance(data, np.ndarray):
        print(f"불러온 NumPy 배열의 형태(shape): {data.shape}")
        print(f"불러온 NumPy 배열의 dtype: {data.dtype}")

    # shape가 ()인 것은 배열이 0차원(스칼라)임을 의미합니다.
    # 이는 보통 실제 데이터가 단일 객체로 저장되었을 때 발생합니다.
    if isinstance(data, np.ndarray) and data.shape == ():
        # .item() 메서드를 사용하여 배열 안의 실제 객체를 추출합니다.
        actual_data = data.item()
        print(f"추출된 실제 데이터의 타입: {type(actual_data)}")

        # 추출된 데이터가 딕셔너리일 경우, 내부 구조를 상세히 분석합니다.
        if isinstance(actual_data, dict):
            print("\n--- 딕셔너리 내용물 상세 분석 ---")
            print(f"딕셔너리의 키(keys): {list(actual_data.keys())}")

            for key, value in actual_data.items():
                print(f"\n  - 키: '{key}'")
                print(f"    - 값의 타입: {type(value)}")

                # 값이 NumPy 배열이면 형태/타입과 함께 값 미리보기를 출력합니다.
                if isinstance(value, np.ndarray):
                    print(f"    - 값(NumPy 배열) shape: {value.shape}, dtype: {value.dtype}")
                    print(f"    - 값 미리보기: {_preview_ndarray(value)}")
                # 값이 리스트나 튜플이면 길이와 일부 샘플을 출력합니다.
                elif isinstance(value, (list, tuple)):
                    print(f"    - 값(리스트/튜플) 길이: {len(value)}")
                    sample = value[:10] if isinstance(value, (list, tuple)) else []
                    print(f"    - 앞 10개 샘플: {_safe_repr(sample)}")
                # 그 외의 경우, 값을 출력합니다. (너무 길면 일부만 표시)
                else:
                    print(f"    - 값: {_safe_repr(value)}")
        else:
            # 딕셔너리가 아닐 경우도 값 자체를 보여줍니다.
            if isinstance(actual_data, np.ndarray):
                print(f"실제 데이터(NumPy 배열) shape: {actual_data.shape}, dtype: {actual_data.dtype}")
                print(f"값 미리보기: {_preview_ndarray(actual_data)}")
            else:
                print(f"실제 데이터 값: {_safe_repr(actual_data)}")

    else:
        # 0차원이 아닌 일반 NumPy 배열일 때, 값 미리보기 출력
        if isinstance(data, np.ndarray):
            print("\n--- 배열 값 미리보기 ---")
            print(_preview_ndarray(data))
            # 수치형이면 추가 요약정보 출력 시도
            try:
                if np.issubdtype(data.dtype, np.number) and data.size > 0:
                    print(f"최솟값: {np.min(data)}, 최댓값: {np.max(data)}")
                    print(f"평균: {np.mean(data):.6f}, 표준편차: {np.std(data):.6f}")
            except Exception:
                pass
        else:
            # np.load 결과가 ndarray가 아닌 특수 케이스(희박) 처리
            print(f"불러온 데이터: {_safe_repr(data)}")

except FileNotFoundError:
    print(f"오류: '{file_path}' 경로에 파일이 존재하지 않습니다.")
except Exception as e:
    print(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
