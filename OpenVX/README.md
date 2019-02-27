# 车道线检测实例
执行目录下脚本
```
bash script/OpenVX_s/lane_detection.sh
```
输出如下：
```
Build Source                                                
-- lane detection alone                                     
-- The C compiler identification is GNU 5.4.0               
-- The CXX compiler identification is GNU 5.4.0             
-- Check for working C compiler: /usr/bin/cc                
-- Check for working C compiler: /usr/bin/cc -- works       
-- Detecting C compiler ABI info                            
-- Detecting C compiler ABI info - done                     
-- Detecting C compile features                             
-- Detecting C compile features - done                      
-- Check for working CXX compiler: /usr/bin/c++             
-- Check for working CXX compiler: /usr/bin/c++ -- works    
-- Detecting CXX compiler ABI info                          
-- Detecting CXX compiler ABI info - done                   
.....
Input frame size: 960x508                                                                                 
Warp Perspective Matrix = 9::9.381812,5.609933,0.019545,3.800000,0.000000,0.000000,24.000006,421.639984,1.
000000                                                                                                    
Frame: 300                                                                                                
Release data...                                                                                           
2.526 ms by ReadFrame averaged by 301 samples                                                             
3.280 ms by ProcessOpenCVReference averaged by 300 samples                                                
2.498 ms by vxProcessGraph averaged by 300 samples                                                        
0.410 ms by CollectLaneMarks averaged by 600 samples                                                      

```
