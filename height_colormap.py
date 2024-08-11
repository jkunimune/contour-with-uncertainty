"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license,  visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
from matplotlib.colors import ListedColormap

cm_data = [[0, 0, 0],
           [4.16093296e-05, 1.76325626e-03, 1.03123230e-03],
           [4.99578860e-04, 2.90134154e-03, 2.10444237e-03],
           [1.14385912e-03, 4.28090879e-03, 3.52074067e-03],
           [1.95836306e-03, 5.89465397e-03, 5.27862340e-03],
           [2.93018980e-03, 7.73759583e-03, 7.37842620e-03],
           [4.04817570e-03, 9.80614220e-03, 9.82371199e-03],
           [5.30248571e-03, 1.20975527e-02, 1.26208367e-02],
           [6.68497421e-03, 1.46095582e-02, 1.57768688e-02],
           [8.18834458e-03, 1.73402982e-02, 1.93006147e-02],
           [9.80659611e-03, 2.02881407e-02, 2.32006872e-02],
           [1.15331314e-02, 2.34519098e-02, 2.74892765e-02],
           [1.33641662e-02, 2.68302844e-02, 3.21736872e-02],
           [1.52942903e-02, 3.04224691e-02, 3.72663487e-02],
           [1.73192171e-02, 3.42277174e-02, 4.27017165e-02],
           [1.94353663e-02, 3.82453722e-02, 4.81417646e-02],
           [2.16387659e-02, 4.24126574e-02, 5.35699903e-02],
           [2.39260025e-02, 4.65339557e-02, 5.89899578e-02],
           [2.62935318e-02, 5.06140790e-02, 6.44052561e-02],
           [2.87277889e-02, 5.46601980e-02, 6.98108813e-02],
           [3.11937827e-02, 5.86841750e-02, 7.52094799e-02],
           [3.36770285e-02, 6.26899525e-02, 8.06031244e-02],
           [3.61642325e-02, 6.66808746e-02, 8.59916567e-02],
           [3.86410813e-02, 7.06602446e-02, 9.13742966e-02],
           [4.10788240e-02, 7.46313551e-02, 9.67487946e-02],
           [4.33877388e-02, 7.85975074e-02, 1.02112053e-01],
           [4.55708741e-02, 8.25615514e-02, 1.07461500e-01],
           [4.76244867e-02, 8.65262649e-02, 1.12793339e-01],
           [4.95449748e-02, 9.04942089e-02, 1.18103124e-01],
           [5.13287605e-02, 9.44677365e-02, 1.23386053e-01],
           [5.29715346e-02, 9.84494467e-02, 1.28634924e-01],
           [5.44710051e-02, 1.02441045e-01, 1.33843928e-01],
           [5.58265699e-02, 1.06443471e-01, 1.39008595e-01],
           [5.70369733e-02, 1.10457928e-01, 1.44122344e-01],
           [5.81002370e-02, 1.14485854e-01, 1.49176792e-01],
           [5.90197243e-02, 1.18526521e-01, 1.54170021e-01],
           [5.97970529e-02, 1.22579866e-01, 1.59097492e-01],
           [6.04313132e-02, 1.26646646e-01, 1.63951877e-01],
           [6.09277746e-02, 1.30725218e-01, 1.68733873e-01],
           [6.12889268e-02, 1.34814833e-01, 1.73441786e-01],
           [6.15168387e-02, 1.38914720e-01, 1.78074632e-01],
           [6.16112142e-02, 1.43024699e-01, 1.82630213e-01],
           [6.15755173e-02, 1.47143223e-01, 1.87111220e-01],
           [6.14101036e-02, 1.51269482e-01, 1.91519227e-01],
           [6.11151106e-02, 1.55402672e-01, 1.95855839e-01],
           [6.06887019e-02, 1.59542228e-01, 2.00123257e-01],
           [6.01290705e-02, 1.63687506e-01, 2.04323780e-01],
           [5.94331075e-02, 1.67837982e-01, 2.08459965e-01],
           [5.85964569e-02, 1.71993225e-01, 2.12534511e-01],
           [5.76297634e-02, 1.76152386e-01, 2.16534769e-01],
           [5.65526333e-02, 1.80315346e-01, 2.20433398e-01],
           [5.53760175e-02, 1.84480952e-01, 2.24224174e-01],
           [5.41089993e-02, 1.88647731e-01, 2.27907262e-01],
           [5.27645714e-02, 1.92814062e-01, 2.31481471e-01],
           [5.13571098e-02, 1.96978307e-01, 2.34946421e-01],
           [4.99015462e-02, 2.01138897e-01, 2.38302857e-01],
           [4.84130040e-02, 2.05294375e-01, 2.41552573e-01],
           [4.69069345e-02, 2.09443389e-01, 2.44698328e-01],
           [4.54012453e-02, 2.13584725e-01, 2.47741564e-01],
           [4.39129705e-02, 2.17717435e-01, 2.50684326e-01],
           [4.24499303e-02, 2.21840943e-01, 2.53534708e-01],
           [4.10249239e-02, 2.25954638e-01, 2.56297776e-01],
           [3.96420290e-02, 2.30058277e-01, 2.58976247e-01],
           [3.83456594e-02, 2.34151324e-01, 2.61570442e-01],
           [3.71392523e-02, 2.38234173e-01, 2.64092050e-01],
           [3.60324087e-02, 2.42306826e-01, 2.66542119e-01],
           [3.50348307e-02, 2.46369303e-01, 2.68921501e-01],
           [3.41320040e-02, 2.50422254e-01, 2.71242555e-01],
           [3.33495993e-02, 2.54465509e-01, 2.73496037e-01],
           [3.26604383e-02, 2.58500069e-01, 2.75699241e-01],
           [3.20900915e-02, 2.62525861e-01, 2.77841874e-01],
           [3.16120122e-02, 2.66543909e-01, 2.79939403e-01],
           [3.12492657e-02, 2.70554255e-01, 2.81981926e-01],
           [3.09816393e-02, 2.74557717e-01, 2.83981531e-01],
           [3.08145472e-02, 2.78554671e-01, 2.85936685e-01],
           [3.07512322e-02, 2.82545628e-01, 2.87845859e-01],
           [3.07783969e-02, 2.86531170e-01, 2.89717435e-01],
           [3.09004278e-02, 2.90511769e-01, 2.91548881e-01],
           [3.11209749e-02, 2.94487846e-01, 2.93338485e-01],
           [3.14288761e-02, 2.98459863e-01, 2.95093469e-01],
           [3.18207894e-02, 3.02428461e-01, 2.96813804e-01],
           [3.23020684e-02, 3.06393935e-01, 2.98497277e-01],
           [3.28728906e-02, 3.10356673e-01, 3.00143847e-01],
           [3.35244893e-02, 3.14317165e-01, 3.01757811e-01],
           [3.42552680e-02, 3.18275843e-01, 3.03339299e-01],
           [3.50659876e-02, 3.22232926e-01, 3.04889041e-01],
           [3.59541386e-02, 3.26188904e-01, 3.06406797e-01],
           [3.69216514e-02, 3.30144034e-01, 3.07891942e-01],
           [3.79702144e-02, 3.34098615e-01, 3.09343436e-01],
           [3.90932030e-02, 3.38053060e-01, 3.10763833e-01],
           [4.02903265e-02, 3.42007642e-01, 3.12153271e-01],
           [4.15334629e-02, 3.45962668e-01, 3.13511678e-01],
           [4.28241064e-02, 3.49918251e-01, 3.14839206e-01],
           [4.41676738e-02, 3.53874334e-01, 3.16135659e-01],
           [4.55643726e-02, 3.57831034e-01, 3.17400495e-01],
           [4.70153662e-02, 3.61788372e-01, 3.18633522e-01],
           [4.85220875e-02, 3.65746333e-01, 3.19834697e-01],
           [5.00848758e-02, 3.69704984e-01, 3.21003637e-01],
           [5.17043873e-02, 3.73664353e-01, 3.22140113e-01],
           [5.33818155e-02, 3.77624404e-01, 3.23244161e-01],
           [5.51172525e-02, 3.81585189e-01, 3.24315442e-01],
           [5.69108842e-02, 3.85546740e-01, 3.25353687e-01],
           [5.87634104e-02, 3.89509030e-01, 3.26358905e-01],
           [6.06748259e-02, 3.93472085e-01, 3.27330850e-01],
           [6.26449072e-02, 3.97435945e-01, 3.28269212e-01],
           [6.46740874e-02, 4.01400572e-01, 3.29174045e-01],
           [6.67619650e-02, 4.05366003e-01, 3.30045043e-01],
           [6.89081241e-02, 4.09332272e-01, 3.30881915e-01],
           [7.11130017e-02, 4.13299308e-01, 3.31684888e-01],
           [7.33754846e-02, 4.17267204e-01, 3.32453360e-01],
           [7.56955649e-02, 4.21235924e-01, 3.33187389e-01],
           [7.80728189e-02, 4.25205471e-01, 3.33886833e-01],
           [8.05064101e-02, 4.29175892e-01, 3.34551301e-01],
           [8.29964096e-02, 4.33147121e-01, 3.35181051e-01],
           [8.55415459e-02, 4.37119250e-01, 3.35775416e-01],
           [8.81418354e-02, 4.41092208e-01, 3.36334695e-01],
           [9.07961816e-02, 4.45066064e-01, 3.36858353e-01],
           [9.35043151e-02, 4.49040773e-01, 3.37346536e-01],
           [9.62652919e-02, 4.53016383e-01, 3.37798830e-01],
           [9.90786396e-02, 4.56992873e-01, 3.38215250e-01],
           [1.01943566e-01, 4.60970268e-01, 3.38595517e-01],
           [1.04859434e-01, 4.64948568e-01, 3.38939507e-01],
           [1.07825571e-01, 4.68927781e-01, 3.39247060e-01],
           [1.10841317e-01, 4.72907911e-01, 3.39518036e-01],
           [1.13905890e-01, 4.76888985e-01, 3.39752127e-01],
           [1.17018812e-01, 4.80870978e-01, 3.39949423e-01],
           [1.20179149e-01, 4.84853949e-01, 3.40109368e-01],
           [1.23386554e-01, 4.88837850e-01, 3.40232251e-01],
           [1.26639997e-01, 4.92822767e-01, 3.40317271e-01],
           [1.29939295e-01, 4.96808619e-01, 3.40365032e-01],
           [1.33286662e-01, 5.00795267e-01, 3.40373876e-01],
           [1.36681153e-01, 5.04782704e-01, 3.40344153e-01],
           [1.40119568e-01, 5.08771132e-01, 3.40276181e-01],
           [1.43601108e-01, 5.12760614e-01, 3.40169291e-01],
           [1.47125526e-01, 5.16751082e-01, 3.40024077e-01],
           [1.50691944e-01, 5.20742631e-01, 3.39839549e-01],
           [1.54299923e-01, 5.24735248e-01, 3.39615768e-01],
           [1.57948981e-01, 5.28728933e-01, 3.39352710e-01],
           [1.61638451e-01, 5.32723747e-01, 3.39049660e-01],
           [1.65367939e-01, 5.36719671e-01, 3.38706804e-01],
           [1.69136923e-01, 5.40716730e-01, 3.38323842e-01],
           [1.72949748e-01, 5.44714556e-01, 3.37898763e-01],
           [1.76803934e-01, 5.48713304e-01, 3.37432161e-01],
           [1.80696267e-01, 5.52713220e-01, 3.36924656e-01],
           [1.84626218e-01, 5.56714366e-01, 3.36375459e-01],
           [1.88593363e-01, 5.60716767e-01, 3.35784256e-01],
           [1.92597352e-01, 5.64720397e-01, 3.35151431e-01],
           [1.96637699e-01, 5.68725333e-01, 3.34475955e-01],
           [2.00714016e-01, 5.72731601e-01, 3.33757485e-01],
           [2.04830483e-01, 5.76738776e-01, 3.32994442e-01],
           [2.08986420e-01, 5.80746883e-01, 3.32186719e-01],
           [2.13177236e-01, 5.84756385e-01, 3.31334755e-01],
           [2.17402586e-01, 5.88767310e-01, 3.30438136e-01],
           [2.21662134e-01, 5.92779689e-01, 3.29496454e-01],
           [2.25955526e-01, 5.96793526e-01, 3.28509857e-01],
           [2.30282456e-01, 6.00808888e-01, 3.27477239e-01],
           [2.34649456e-01, 6.04825081e-01, 3.26395672e-01],
           [2.39052095e-01, 6.08842557e-01, 3.25266008e-01],
           [2.43487398e-01, 6.12861636e-01, 3.24088653e-01],
           [2.47955097e-01, 6.16882358e-01, 3.22862929e-01],
           [2.52472692e-01, 6.20902911e-01, 3.21578590e-01],
           [2.57072579e-01, 6.24918376e-01, 3.20244624e-01],
           [2.61766066e-01, 6.28926704e-01, 3.18865807e-01],
           [2.66551439e-01, 6.32927843e-01, 3.17441212e-01],
           [2.71421846e-01, 6.36922678e-01, 3.15965409e-01],
           [2.76378197e-01, 6.40910691e-01, 3.14439940e-01],
           [2.81421472e-01, 6.44891398e-01, 3.12865295e-01],
           [2.86552761e-01, 6.48864318e-01, 3.11241229e-01],
           [2.91773477e-01, 6.52828965e-01, 3.09566297e-01],
           [2.97083918e-01, 6.56784813e-01, 3.07843031e-01],
           [3.02490638e-01, 6.60730359e-01, 3.06074648e-01],
           [3.07993084e-01, 6.64665365e-01, 3.04259180e-01],
           [3.13586403e-01, 6.68590174e-01, 3.02396351e-01],
           [3.19271712e-01, 6.72504258e-01, 3.00485104e-01],
           [3.25049300e-01, 6.76407101e-01, 2.98526768e-01],
           [3.30919574e-01, 6.80298183e-01, 2.96522086e-01],
           [3.36883267e-01, 6.84176962e-01, 2.94470836e-01],
           [3.42939981e-01, 6.88042956e-01, 2.92375510e-01],
           [3.49090716e-01, 6.91895597e-01, 2.90235007e-01],
           [3.55334695e-01, 6.95734434e-01, 2.88052333e-01],
           [3.61672485e-01, 6.99558919e-01, 2.85827215e-01],
           [3.68103490e-01, 7.03368596e-01, 2.83561874e-01],
           [3.74627573e-01, 7.07162977e-01, 2.81257367e-01],
           [3.81244491e-01, 7.10941581e-01, 2.78914892e-01],
           [3.87953148e-01, 7.14704021e-01, 2.76537226e-01],
           [3.94753717e-01, 7.18449784e-01, 2.74124535e-01],
           [4.01644630e-01, 7.22178552e-01, 2.71680271e-01],
           [4.08625011e-01, 7.25889944e-01, 2.69206408e-01],
           [4.15694404e-01, 7.29583537e-01, 2.66704103e-01],
           [4.22850828e-01, 7.33259109e-01, 2.64177198e-01],
           [4.30092950e-01, 7.36916381e-01, 2.61628245e-01],
           [4.37420011e-01, 7.40555004e-01, 2.59058771e-01],
           [4.44829620e-01, 7.44174878e-01, 2.56473059e-01],
           [4.52320064e-01, 7.47775832e-01, 2.53874147e-01],
           [4.59889518e-01, 7.51357736e-01, 2.51265233e-01],
           [4.67536256e-01, 7.54920466e-01, 2.48649332e-01],
           [4.75257676e-01, 7.58464066e-01, 2.46030860e-01],
           [4.83051589e-01, 7.61988548e-01, 2.43413540e-01],
           [4.90915663e-01, 7.65493975e-01, 2.40801322e-01],
           [4.98847627e-01, 7.68980436e-01, 2.38198082e-01],
           [5.06846674e-01, 7.72447606e-01, 2.35611161e-01],
           [5.14911013e-01, 7.75895463e-01, 2.33046192e-01],
           [5.23035198e-01, 7.79325023e-01, 2.30503919e-01],
           [5.31216504e-01, 7.82736607e-01, 2.27988941e-01],
           [5.39452169e-01, 7.86130582e-01, 2.25505959e-01],
           [5.47739460e-01, 7.89507351e-01, 2.23059683e-01],
           [5.56075703e-01, 7.92867343e-01, 2.20654797e-01],
           [5.64458216e-01, 7.96211031e-01, 2.18296052e-01],
           [5.72884063e-01, 7.99538983e-01, 2.15988609e-01],
           [5.81350805e-01, 8.02851699e-01, 2.13736987e-01],
           [5.89855957e-01, 8.06149729e-01, 2.11545830e-01],
           [5.98397138e-01, 8.09433634e-01, 2.09419683e-01],
           [6.06971276e-01, 8.12704176e-01, 2.07364057e-01],
           [6.15576638e-01, 8.15961843e-01, 2.05382689e-01],
           [6.24211044e-01, 8.19207257e-01, 2.03479944e-01],
           [6.32872063e-01, 8.22441137e-01, 2.01660515e-01],
           [6.41557501e-01, 8.25664180e-01, 1.99928764e-01],
           [6.50266126e-01, 8.28876869e-01, 1.98287794e-01],
           [6.58995482e-01, 8.32080025e-01, 1.96742238e-01],
           [6.67743991e-01, 8.35274275e-01, 1.95295539e-01],
           [6.76510774e-01, 8.38460085e-01, 1.93950206e-01],
           [6.85292964e-01, 8.41638471e-01, 1.92711106e-01],
           [6.94090497e-01, 8.44809724e-01, 1.91579542e-01],
           [7.02901682e-01, 8.47974588e-01, 1.90558690e-01],
           [7.11749019e-01, 8.51126959e-01, 1.89623264e-01],
           [7.20593798e-01, 8.54281850e-01, 1.88648196e-01],
           [7.29437962e-01, 8.57439480e-01, 1.87613506e-01],
           [7.38278676e-01, 8.60601039e-01, 1.86522865e-01],
           [7.47118326e-01, 8.63766207e-01, 1.85373588e-01],
           [7.55957457e-01, 8.66935184e-01, 1.84165115e-01],
           [7.64796783e-01, 8.70108116e-01, 1.82896591e-01],
           [7.73638822e-01, 8.73284587e-01, 1.81564603e-01],
           [7.82485690e-01, 8.76464280e-01, 1.80165967e-01],
           [7.91339406e-01, 8.79646880e-01, 1.78697336e-01],
           [8.00201817e-01, 8.82832108e-01, 1.77155297e-01],
           [8.09075252e-01, 8.86019509e-01, 1.75535400e-01],
           [8.17951821e-01, 8.89211878e-01, 1.73848337e-01],
           [8.26836277e-01, 8.92408015e-01, 1.72086102e-01],
           [8.35737091e-01, 8.95605454e-01, 1.70233985e-01],
           [8.44654707e-01, 8.98804276e-01, 1.68288959e-01],
           [8.53572952e-01, 9.02010075e-01, 1.66275531e-01],
           [8.62516040e-01, 9.05215107e-01, 1.64151366e-01],
           [8.71467960e-01, 9.08424941e-01, 1.61940583e-01],
           [8.80440943e-01, 9.11635709e-01, 1.59618836e-01],
           [8.89426025e-01, 9.14850690e-01, 1.57198474e-01],
           [8.98439159e-01, 9.18064629e-01, 1.54645742e-01],
           [9.07459820e-01, 9.21284845e-01, 1.51994547e-01],
           [9.16503465e-01, 9.24506212e-01, 1.49210193e-01],
           [9.25570217e-01, 9.27728877e-01, 1.46285669e-01],
           [9.34660937e-01, 9.30952725e-01, 1.43211507e-01],
           [9.43776608e-01, 9.34177589e-01, 1.39976734e-01],
           [9.52919635e-01, 9.37402777e-01, 1.36565605e-01],
           [9.62085050e-01, 9.40630279e-01, 1.32977608e-01],
           [9.71285495e-01, 9.43855628e-01, 1.29168740e-01],
           [9.80520671e-01, 9.47079040e-01, 1.25121094e-01],
           [9.89815547e-01, 9.50291239e-01, 1.20745996e-01],
           [9.99294958e-01, 9.53444802e-01, 1.15657721e-01]]
height_colormap = ListedColormap(cm_data)