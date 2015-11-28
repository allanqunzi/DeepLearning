local StockDataLoader = {}
StockDataLoader.__index = StockDataLoader
-- csv_dir should be ended with '/'
function StockDataLoader.create(csv_dir, row_num, col_num)
    local self = {}
    setmetatable(self, StockDataLoader)
    local csv_filename = csv_dir..'AAPL_10years_HistoricalQuotes.csv'
    self.row_num = row_num
    self.col_num = col_num

    local infile = io.open(csv_filename, 'r')
    local header = infile:read()
    infile:read() -- blank line
    self.data = torch.Tensor(row_num, col_num)

    local i = 0
    local j = -1
    for line in infile:lines('*l') do
        i = i + 1
        j = -1
        for item in string.gmatch(line,'[^",]+') do
            j = j + 1
            if j > 0 and j < 6 then
                self.data[i][j] = tonumber(item)
            end
        end
    end
    infile:close()
    torch.save(csv_dir..'aapl.dat', self.data) -- , 'ascii')
    return self
end

-- exponential moving average
function StockDataLoader.ema(period)
    local alpha = 2.0/(period + 1) -- decay percentage
    assert(self.row_num == self.data:size(1))
    local ema_data = torch.zeros(self.row_num)
    ema_data[1] = self.data[1][1]
    for i=2, self.row_num do
        ema_data[i] = ema_data[i-1] + alpha * (self.data[i][1] - ema_data[i-1])
    end
    return ema_data
end

-- simple moving average with +/-2*sigma
function StockDataLoader.sma(period)
    local data_size = self.col_num - period
    local sma_data = torch.Tensor(data_size, 3)
    local multiplier = 1.0/period
    local avg = 0.0
    local avgsq = 0.0
    local head = self.data[1][1]
    local headsq = head * head
    for i=1, period do
        avg = avg + self.data[i][1]
        avgsq = avgsq + self.data[i][1] * self.data[i][1]
    end
    local tmp1
    local tmp2
    for i=(period+1), self.col_num do
        tmp1 = avg * multiplier
        tmp2 = avgsq * multiplier
        tmp2 = math.sqrt(tmp2 - tmp1 * tmp1)
        sma_data[i-period][1] = tmp1
        sma_data[i-period][2] = tmp1 - 2.0 * tmp2
        sma_data[i-period][3] = tmp1 + 2.0 * tmp2
        avg = avg - head + self.data[i][1]
        avgsq = avg - headsq + self.data[i][1] * self.data[i][1]
        head = self.data[i-period+1][1]
        headsq = head * head
    end
    return sma_data
end

return StockDataLoader
