setwd("C:\\Users\\Yao\\Downloads\\YelpData")
library(ggmap)
geocodeQueryCheck()
origAddress <- read.csv("origAddress.csv", stringsAsFactors = FALSE)
summary(origAddress)
# Loop through the addresses to get the latitude and longitude of each address and add it to the
# origAddress data frame in new columns lat and lon
for(i in 1:nrow(origAddress)) {
  result <- tryCatch(geocode(origAddress$addresses[i], output = "latlona", source = "google"),
                     warning = function(w) data.frame(lon = NA, lat = NA, address = NA))
  origAddress$lon[i] <- as.numeric(result[1])
  origAddress$lat[i] <- as.numeric(result[2])
  origAddress$geoAddress[i] <- as.character(result[3])
}
# Write a CSV file containing origAddress to the working directory
write.csv(origAddress, "geocoded.csv", row.names=FALSE)
