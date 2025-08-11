import os

folder = '../DATASETS/VISDA-C'  # change to your path
domains = os.listdir(folder)
domains.sort()

for d in range(len(domains)):
    dom = domains[d]
    if os.path.isdir(os.path.join(folder, dom)):
        dom_new = dom.replace(" ", "_")
        # print(dom, dom_new)
        os.rename(os.path.join(folder, dom), os.path.join(folder, dom_new))

        classes = os.listdir(os.path.join(folder, dom_new))
        classes.sort()
        f = open(dom_new + "_list.txt", "w")
        for c in range(len(classes)):
            print("{}:{}".format(classes[c], c))
            cla = classes[c]
            cla_new = cla.replace(" ", "_")
            # print(cla, cla_new)
            os.rename(os.path.join(folder, dom_new, cla), os.path.join(folder, dom_new, cla_new))
            files = os.listdir(os.path.join(folder, dom_new, cla_new))
            files.sort()
            for file in files:
                file_new = file.replace(" ", "_")
                # rename
                os.rename(os.path.join(folder, dom_new, cla_new, file),
                          os.path.join(folder, dom_new, cla_new, file_new))

                # use '/' to separate
                file_path = os.path.join(folder, dom_new, cla_new, file_new)
                file_path = file_path.replace(os.sep, '/')  # force `/`
                # print(file, file_new)
                # print('{:} {:}'.format(file_path, c))
                f.write('{:} {:}\n'.format(file_path, c))
        f.close()
