require 'csv'

CSV.foreach(ARGV[0]) do |row|
  taxID = row[0]
  return_period = row[1]
  name = row[2]
  state = row[3]
  zipcode = row[4]
  specified_return = row[5]
  net_assets = row[7]
  date = row[8]

  row[9] =~ /[A-Z]+:\\(.+)/
  puts $1
end
