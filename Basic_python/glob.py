import glob
'''
globモジュールでは、引数に指定されたパターンにマッチするファイルパス名を取得することが出来ます.
特定のディレクトリに存在するファイルに処理を加えたい場合などに、使用します.
マッチングさせるパターンの書き方は、Unixシェルで使用される書き方と同じです.
正規表現，パターンマッチ
'''

py_files = glob.glob("numpy/*py")
print(py_files)
for file in glob.glob("*py"):
    print(file)