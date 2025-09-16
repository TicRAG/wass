import wrench

print("--- WRENCH Module API ---")
# 打印出 wrench 模块所有可用的、非内部的属性（主要是类和函数）
count = 0
for item in dir(wrench):
    if not item.startswith("_"): 
        print(item)
        count += 1
print("----------------------------")
print(f"Found {count} public attributes/classes in the 'wrench' module.")