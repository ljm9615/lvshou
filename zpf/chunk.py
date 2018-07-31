import feather
def gen_csv_feather(path,path_new):
    f = open(path)
    reader = pd.read_csv(f, sep=',', iterator=True)
    loop = True
    chunkSize = 10000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)
    print(df.count())
    feather.write_dataframe(df,path_new)
    
gen_csv_feather("/home/scw4150/Desktop/new_data/data/train_set.csv","/home/scw4150/Desktop/new_data/data/train_set.feather")
gen_csv_feather("/home/scw4150/Desktop/new_data/data/test_set.csv","/home/scw4150/Desktop/new_data/data/test_set.feather")

train=feather.read_dataframe("/home/scw4150/Desktop/new_data/data/train_set.feather")
test=feather.read_dataframe("/home/scw4150/Desktop/new_data/data/test_set.feather")