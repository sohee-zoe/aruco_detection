import yaml
import numpy as np

def save_calibration_to_yaml(filename, K, D):
    """카메라 매트릭스(K)와 왜곡 계수(D)를 YAML 파일에 저장합니다."""
    data = {
        "K": K.tolist(),
        "D": D.tolist()
    }
    with open(filename, 'w') as f:
        yaml.dump(data, f)

def load_calibration_from_yaml(filename):
    """YAML 파일에서 카메라 매트릭스(K)와 왜곡 계수(D)를 로드합니다."""
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data["K"])
    D = np.array(data["D"])
    return K, D
