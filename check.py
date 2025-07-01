# Giả sử bạn có đoạn văn nằm trong 1 danh sách (list) như sau:
data = ["in\tgeneral\tconstitutes\ta\tmajor\tuse\tof\tJava.\nThe\tJava\tBuzzwords\n...",
        "CHAPTER 1  All about Java      15\ncontrol handheld devices, and more...",
        "that\t\ncould\tbe\tused\tto\tproduce\tcode\tthat\twould\trun\ton\ta\tvariety\tof\tCPUs..."]

# Lọc bỏ tất cả ký tự tab (\t) và xuống dòng (\n)
cleaned_data = [item.replace('\t', ' ').replace('\n', ' ') for item in data]

# Nếu muốn gộp tất cả thành 1 chuỗi duy nhất:
final_text = ' '.join(cleaned_data)

# In ra kết quả cuối
print(final_text)
