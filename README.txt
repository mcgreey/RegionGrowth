功能：
区域生长算法，用于二维图像的分割，使用时将regiongrow函数直接扣出来，放自己代码里

主要函数为seg_image, num_region = regiongrow(orig_image, seed_matrix, threshold)

使用说明
形参：
1.orig_image：需转化为灰度值范围的orig_image
2.seed_matrix：size与orig_image相同，需自己看原图进行指定
3.threshold：阈值，需提前查看目标灰度值与周边的差值

输出：
1.seg_image：分割后的mask
2.num_region：分割连通区域的个数