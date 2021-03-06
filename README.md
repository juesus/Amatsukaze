# Amatsukaze
Automated MPEG2-TS Transcoder

# これは何？
TSファイルをエンコしてmp4にするだけの単機能プログラム。

# 特徴
なるべく元ファイルのフォーマットをそのままmp4に変換する。インタレ保持、RFFフラグ保持、音声は無変換でmp4化する。

普通のエンコーダと最も違う点は、1ファイル入力→複数ファイル出力である点。Amatsukazeは単一出力ファイル単一フォーマットとなるように出力するため、入力ファイルによっては複数のmp4ファイルが出力されることがある。

# なぜ作った？
既存ソフトは品質が悪いから

例）

- RFFフラグをきちんと処理できていない
- 5.1ch音声と2ch音声の混ざったmp4を平気で出力する
- 音ズレする
- 自動化できない など

目指したのは**エンコード後の確認は不要**とすること。原理的に絶対に音ズレしない仕組みを導入。エラーは誤魔化さずにエラーとし、中途半端なファイルは作らないよう細心の注意を払った。

# 対応入力フォーマット

- コンテナ: MPEG2-TS 188バイトパケット
- 映像: MPEG2, H264
- 音声: MPEG2-AAC

**これだけ！**

# 対応エンコーダ
x264, x265, QSVEnc

# 注意事項
- RFFフラグ保持でエンコードするには改造x264が必要
- H265のインタレ動画は通常のPC環境では再生できないので実用的でない

# Amatsukaze.exe
Amatsukaze.exeは入力ファイルを解析し実際のエンコードを行うCUIプログラム。

```
Amatsukaze.exe <オプション> -i <input.ts> -o <output.mp4>
オプション []はデフォルト値
  -i|--input  <パス>  入力ファイルパス
  -o|--output <パス>  出力ファイルパス
  --mode <モード>     処理モード[ts]
                      ts : MPGE2-TSを入力する詳細解析モード
                      g  : FFMPEGを利用した一般ファイルモード
  -s|--serviceid <数値> 処理するサービスIDを指定[]
  -w|--work   <パス>  一時ファイルパス[./]
  -et|--encoder-type <タイプ>  使用エンコーダタイプ[x264]
                      対応エンコーダ: x264,x265,QSVEnc
  -e|--encoder <パス> エンコーダパス[x264.exe]
  -eo|--encoder-opotion <オプション> エンコーダへ渡すオプション[]
                      入力ファイルの解像度、アスペクト比、インタレースフラグ、
                      フレームレート、カラーマトリクス等は自動で追加されるので不要
  -b|--bitrate a:b:f  ビットレート計算式 映像ビットレートkbps = f*(a*s+b)
                      sは入力映像ビットレート、fは入力がH264の場合は入力されたfだが、
                      入力がMPEG2の場合はf=1とする
                      指定がない場合はビットレートオプションを追加しない
  --2pass             2passエンコード
  --pulldown          ソフトテレシネを解除しないでそのままエンコード
                      エンコーダの--pdfile-inオプションへの対応が必要
  -m|--muxer  <パス>  L-SMASHのmuxerへのパス[muxer.exe]
  -t|--timelineeditor <パス> L-SMASHのtimelineeditorへのパス[timelineeditor.exe]
  -j|--json   <パス>  出力結果情報をJSON出力する場合は出力ファイルパスを指定[]
  --dump              処理途中のデータをダンプ（デバッグ用）
  ```
  
※一般ファイルモードはTS以外のファイルも読めるようにしたオマケ機能です。普通にFFMPEGを使ってデコードし、音ズレ修正など一切行わないので、TSファイルに使用すると音ズレその他の不具合が発生します。エンコ済みのmp4を再エンコするとき等に使ってください。
  
# AmatsukazeGUI.exe

AmatsukazeGUI.exeはAmatsukaze.exeを呼び出すGUI。サーバ、クライアント機能があり、クライアントから操作し、サーバでエンコードを実行する。
  
```
  AmatsukazeGUI.exe -l <standalone|server|client>
  オプション
      standalone    サーバ・クライアント一体型で起動
      server        サーバモードで起動
      client        クライアントモードで起動
```

エンコードするには、設定を済ませた後、キューパネルにエンコしたいTSファイルの入っているフォルダをドラッグ＆ドロップする。処理が開始されると、フォルダにencoded,failed,succeededという3つのフォルダが作られ、encodedにエンコード済みmp4ファイル、succeededに成功したTSファイル、failedに失敗したTSファイルが入れられる。

ネットワーク越しのフォルダを追加すると、ネットワーク越しの転送によるデータ化けを防ぐため、入力ファイルのハッシュチェック、出力ファイルのハッシュ生成がONになる。BatchHashChecker（まだ公開してない・・・）でファイルハッシュを生成しておく必要がある。

# ビルド方法
FFmpegのincludeとlibが必要。Windowsビルドのdev版を落として、includeフォルダの中身をプロジェクトのincludeフィルダへコピー、libの中身をプロジェクトのlib/x64(or x86)フォルダへコピーする。

単体テストプロジェクト(AmatsukazeUnitTest)は、他にgoogletestのライブラリが必要。サブモジュールでgoogletestは追加してあるので、git submodule updateでコードを落として、googletest/googletest/msvc/gtest-md.slnを開いて、ビルドする。できたgtest.lib/gtestd.libをlib/x64(or x86)へコピー。

単体テストプロジェクト(AmatsukazeUnitTest)は、他にOpenCVも使っているので、OpenCV 3.2.0をビルドしてこれもlibをlib/x64(or x86)へコピーしよう。

