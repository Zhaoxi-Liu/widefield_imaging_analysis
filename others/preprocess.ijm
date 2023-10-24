//以下3个参数需要修改！！！
path = "Z:/WF_VC_liuzhaoxi/23.9.4_GP4.3/moving-bar/"
datalist = newArray("20230904-215136-470")
cropsize = newArray(150, 150)

print("path: "+path)
print("cropsize: "+cropsize)

inputpath = path+"raw/"
outputPath = path+"process/";
left = cropsize[0];
top = cropsize[1];
width = 256;
height = 256;

for (i = 0; i < datalist.length; i++) {
    datafile=datalist[i];
	//打开tiff stack
    print("start opening "+datafile);
	File.openSequence(inputpath+datafile, "virtual");
	//裁剪图片
	makeRectangle(left, top, width, height);
	print("start cropping "+datafile);
	run("Crop");
	print("finish cropping "+datafile);
	//保存tiff stack		
	saveAs("Tiff", outputPath+datafile+".tif");
	print("finish saving "+datafile);
	
	//导出每帧平均荧光值
	run("Plot Z-axis Profile");
	run("Clear Results");
	Plot.getValues (x, y);		
	for (i = 0; i < x.length; i++)
		setResult ("Mean", i, y[i]);
	updateResults ();
	saveAs("Measurements", outputPath+datafile+"-Values.csv");
	print("finish plotting mean values of "+datafile);
	run("Close");	//关闭Results窗口
	close();	//关闭Z-axis Profile
		
	print(datafile+" finished!\n");
	// close();	//关闭tiff窗口
    }
    
print("All finished!");
