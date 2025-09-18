import os
import json
import numpy as np

def load_known_faces(file_path):
    """
    从 JSON 文件加载人脸编码和名称。
    
    参数:
    file_path (str): JSON 文件的路径。

    返回:
    tuple: 包含人脸编码列表、名称列表和状态消息的元组。
    """
    known_face_encodings = []
    known_face_names = []
    message = "未找到人脸数据。请上传照片。"

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                known_face_encodings = [np.array(e) for e in data.get('encodings', [])]
                known_face_names = data.get('names', [])
            
            num_faces = len(known_face_names)
            if num_faces > 0:
                message = f"已加载 {num_faces} 张已知人脸。准备识别。"
            else:
                message = "人脸数据文件为空。请上传照片。"
        except Exception as e:
            message = f"加载人脸数据时出错：{e}"
    
    return known_face_encodings, known_face_names, message

def save_known_faces(known_face_encodings, known_face_names, file_path):
    """
    将人脸编码和名称保存到 JSON 文件。

    参数:
    known_face_encodings (list): 包含人脸编码的列表。
    known_face_names (list): 包含人脸名称的列表。
    file_path (str): JSON 文件的路径。
    """
    # 将 numpy 数组转换为列表以便 JSON 序列化
    data = {
        'encodings': [e.tolist() for e in known_face_encodings],
        'names': known_face_names
    }
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("人脸数据保存成功。")
    except Exception as e:
        print(f"保存人脸数据时出错：{e}")
