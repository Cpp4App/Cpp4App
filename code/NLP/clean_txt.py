import os

def cleaning_txt(path):
    # f = open("./txt/data_types.txt","r+")
    # f.truncate()
    # g = open("./txt/use_data.txt","r+")
    # g.truncate()
    # e = open("./txt/protect_information.txt","r+")
    # e.truncate()
    # h = open("./txt/children.txt","r+")
    # h.truncate()
    # j = open("./txt/data_retention.txt","r+")
    # j.truncate()
    # k = open("./txt/update.txt","r+")
    # k.truncate()
    # d = open("./txt/region.txt","r+")
    # d.truncate()
    # a = open("./txt/share_information.txt", "r+")
    # a.truncate()
    # b = open("./txt/thrid_party.txt", "r+")
    # b.truncate()
    # c = open("./txt/user_right.txt", "r+")
    # c.truncate()

    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            cleaning_txt(c_path)
        else:
            os.remove(c_path)
    os.removedirs(path)
