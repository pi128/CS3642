from freewili.image import convert
from result import Ok, Err  


img = input("image name no trail: ")

match convert(img + ".png", img + ".fwi"):
    case Ok(msg):
        print(msg)
    case Err(msg):
        print("Error:", msg)