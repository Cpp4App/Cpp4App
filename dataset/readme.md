This file describes the privacy policy dataset, consisting of 30 mobile apps' privacy policies, 230 corresponding original screenshots, 634 labelled screenshots, a table collecting app information and a table containing matching information between the labelled widgets in screenshots and the revelant texts in privacy policies.

The HTML files of privacy policies can be found in file folder 'privacy_policies_html', 
and they are named by the app id recoreded in the table 'app_info.csv'.

The texts of privacy policies can be found in file folder 'privacy_policies', 
and they are named by the app id recoreded in the table 'app_info.csv'.  

The original screenshots of apps  can be found in file folder 'original_screenshots',
and they are named by the app id followed by a symbol '-' and a number.

The labelled screenshots of apps can be found in file folder 'labelled_screenshots',
and they are named by the app id followed by a symbol '-' , the original screenshot number, a symbol '-' and a number.

In the labelled screenshot, the components that may be related to user privacy have been manually framed with red rectangles. The components include texts and icons.

Identifiable personal information in the screenshot has been artificially eliminated.

The location information of widgets can be found in 'location.txt', which contains the id and four indexes on the screenshot: column_min, row_min, column_max, row_max.

In the table 'app_info.csv', the following app information is recorded:
1.id
2.name
3.category
4.downloads+ (number of downloads of this app on Google Play)
5.rating (rating of this app on Google Play)
6.IARC (International Age Rating Coalition)

7.effective date of privacy policy(Format: DD/MM/YYYY)
8.data (date this app information was collected. Format: DD/MM/YYYY)

In the table 'matching.xlsx', the following app information is recorded:

1.id

2.text

"None" value in above two tables means relavent information can not be found.
