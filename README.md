# 3-digit-1-label-classifier
Machine Learning code from university

Input is database of images, each image is a column of 3 handwritten digits from the mnist dataset with a single label:
If digit1<5 then label=digit2 else label=digit3

Standard mnist techniques do not work since the label is not associated with a known digit image, but instead a 3-image set.

I had a dataset of hand-drawn digits (from mnist), but where each image was a column of 3 digits. If the first digit was <5 then the label was for the second digit, otherwise the label was for the third digit. 

To learn it I separated each column into 3 digits, did k-means on the new dataset with a few hundred groups, then assumed the label was correct for both the second and third digit of each column and used this assumption to correctly label each group (50% of the labels would be correct, plus 5% correct by chance, so groups are labelled correctly with high probability). Then to improve accuracy further I made a pruning algorithm to remove groups one at a time in an a* search (remove each group, see which was best, remove it permanently, repeat until accuracy stops increasing). With this I got 88% accuracy in 6 hours training, 92% accuracy in 12 hours training, where the time is based on the number of groups when doing k-means and the accuracy would have gone even higher with more training time.
