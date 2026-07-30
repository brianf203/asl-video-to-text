with open('inference.py', 'r') as f:
    content = f.read()

old = '''    def __getitem__(self, idx):
        return {
            "face_features": torch.tensor(self.face_embeddings),
            "left_hand_features": torch.tensor(self.left_hand_embeddings),
            "right_hand_features": torch.tensor(self.right_hand_embeddings),
            "pose_features": torch.tensor(self.body_posture_embeddings),
        }'''

new = '''    def __getitem__(self, idx):
        return {
            "face_features": torch.tensor(self.face_embeddings).half(),
            "left_hand_features": torch.tensor(self.left_hand_embeddings).half(),
            "right_hand_features": torch.tensor(self.right_hand_embeddings).half(),
            "pose_features": torch.tensor(self.body_posture_embeddings).half(),
        }'''

if old in content:
    content = content.replace(old, new, 1)
    with open('inference.py', 'w') as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Pattern not found")
