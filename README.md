[利用方法]

run.py内に対象画像のディレクトリとlabel pathを記述し、run.pyを実行。


[説明]

学部4年の卒業研究で作成した、単眼RGB画像一枚を入力として、その物体の種類、座標、及び回転角を自作データセットから学習し、推定を行うモジュール。

当時私の所属している研究室ではゴミの分別ロボット製作プロジェクトが行われていました。このモジュールはそのプロジェクトの中の重要な認識部分のモジュールであり、対象物体の種類と位置と回転角を推定するモジュールである。

具体的には、一枚のRGB画像から画像内に映っている物体の物体名と、その物体が存在する座標、及びその物体の正対方向からの回転角を推定する研究である。
さらにそれに加え、カメラの位置毎、つまり対象物体をどの角度から撮影する場合が最も精度が上がるかの比較の研究も行った。
詳細を述べると、原点からの距離を15mで固定し、原点を中心として鉛直方向の面上に描いた円上でカメラを原点方向に向けながら動かすというものである。
初めは一つのネットワークでクラス分類と座標及び回転角を推定する回帰を同時に行うことを試みたが、クラス分類に於いて70%程度の正解率しか出すことができなかった為、
クラス分類と回帰を別々に学習させたところ、カメラと原点とを結ぶ線分が水平方向と60°の角度を成す際に最高精度となり、クラス分類の正解率は86%、回帰の平均誤差は座標(x,y,z方向の誤差の和)が3.7m(カメラによって撮影した入力画像の端から端まではおよそ50m)、角度(これもx,y,z方向の和)が27°を達成することができた。
即ち単純平均すると、x軸方向等、一方向の座標誤差は約1.23m、角度誤差は13.3°を達成することができた。

データセットはUnityで作成した。路上にランダムに車両を配置し、それぞれの物体クラス番号と、その時の座標及び回転角をテキストファイルとして出力し、それを一組として、それぞれのカメラの位置毎に30,000組のデータを作成し、学習を行った。

現在、物体の姿勢認識分野では、推定対象物体の3Dモデルを予めコンピュータ内に保存し、そのモデルとの照合を行うことで位置と角度を推定する研究があるが、それよりも大幅に単純かつ小規模で、動作速度の速いモジュールの製作に成功した。たった二つのネットワークでこれほどの精度を出すことのできた例は少なく、学部生の卒業研究としては良いものとなったように感じている。しかしながら、これに甘んじることなく、今後も能力の向上に向けて精進したい。
