local StockDataLoader = {}
StockDataLoader.__index = StockDataLoader
-- csv_dir should be ended with '/'
function StockDataLoader.create(csv_dir, row_num, col_num)
    local self = {}
    setmetatable(self, StockDataLoader)
    self.row_num = row_num
    self.col_num = col_num
    local raw_stock_data = StockDataLoader.read_csv(csv_dir, 'AAPL_10years_HistoricalQuotes.csv', row_num, col_num)
    local raw_spx_data = StockDataLoader.read_csv(csv_dir, 'SPX_10years_HistoricalQuotes.csv', row_num, col_num)

    self.stock_data = StockDataLoader.reverse(raw_stock_data)
    self.spx_data   = StockDataLoader.reverse(raw_spx_data)

    torch.save(csv_dir..'aapl.dat', self.stock_data, 'ascii')
    torch.save(csv_dir..'spx.dat',  self.spx_data, 'ascii')
    return self
end

function StockDataLoader.read_csv(csv_dir, csv_file, row_num, col_num)
    local csv_filename = csv_dir..csv_file
    local infile = io.open(csv_filename, 'r')
    local header = infile:read()
    infile:read() -- blank line
    local data = torch.Tensor(row_num, col_num)

    local i = 0
    local j = -1
    for line in infile:lines('*l') do
        i = i + 1
        j = -1
        for item in string.gmatch(line,'[^",]+') do
            j = j + 1
            if j > 0 and j < 6 then
                data[i][j] = tonumber(item)
            end
        end
    end
    infile:close()
    return data
end

function StockDataLoader.reverse(data)
    local row_num = data:size(1)
    local col_num = data:size(2)
    local res_data = torch.Tensor(row_num, col_num)
    for i=1, row_num do
        for j=1, col_num do
            res_data[i][j] = data[row_num+1-i][j]
        end
    end
    return res_data
end

function StockDataLoader:create_feature(max_sma_period, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    -- creating features
    assert(self.stock_data:size(1) == self.row_num)
    assert(self.stock_data:size(1) == self.spx_data:size(1))
    self.feature_data = torch.cat(
        self.stock_data:index(2, torch.LongTensor{1,3,4,5}),
        self.spx_data:index(2, torch.LongTensor{1}),
        2)

    -- ema feature
    local ema52 = self:ema(52 * 5)
    assert(self.row_num == ema52:size(1))
    local ema26 = self:ema(26 * 5)
    assert(ema26:size(1) == ema52:size(1))
    local ema_all = torch.cat(ema52, ema26, 2)
    self.feature_data = torch.cat(self.feature_data, ema_all, 2)

    -- sma feature
    local sma10 = self:sma(10)
    assert(self.row_num == sma10:size(1))
    local sma30 = self:sma(30)
    assert(sma30:size(1) == sma10:size(1))
    local sma_all = torch.cat(sma10, sma30, 2)
    self.feature_data = torch.cat(self.feature_data, sma_all, 2)

    local tmp_len = self.row_num - max_sma_period
    if tmp_len % seq_length ~= 0 then
        print('cutting off end of data so that the sequences divide evenly')
        self.feature_data = self.feature_data:sub(max_sma_period+1, seq_length*math.floor(tmp_len/seq_length))
    end
    self.data_len = self.feature_data:size(1)
    self.feature_num = self.feature_data:size(2)
    print('the dimension of the feature data is '..self.data_len..' * '..self.feature_num)

    local ydata = self.feature_data:index(2, torch.LongTensor{1})
    ydata:sub(1,-2):copy(self.feature_data:select(2,1):sub(2,-1))
    self.x_feeds = self.feature_data:split(seq_length, 1)
    self.y_feeds = ydata:split(seq_length, 1)
    assert(#self.x_feeds == #self.y_feeds)
    self.feeds_num = #self.x_feeds

    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1,
        'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1,
        'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    assert(split_fractions[3] >= 0 and split_fractions[3] <= 1,
        'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
    if split_fractions[3] == 0 then
        -- catch a common special case where the user might not want a test set
        self.ntrain = math.floor(self.feeds_num * split_fractions[1])
        self.nval = self.feeds_num - self.ntrain
        self.ntest = 0
    else
        -- divide data to train/val and allocate rest to test
        self.ntrain = math.floor(self.feeds_num * split_fractions[1])
        self.nval = math.floor(self.feeds_num * split_fractions[2])
        self.ntest = self.feeds_num - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    end
    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.feed_ix = {0,0,0}

    print(string.format(
        'data load done. Number of data batches in train: %d, val: %d, test: %d',
        self.ntrain, self.nval, self.ntest))
end


function StockDataLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val', 'test'}
        print('ERROR. Code requested a feed for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.feed_ix[split_index] = self.feed_ix[split_index] + 1
    if self.feed_ix[split_index] > self.split_sizes[split_index] then
        self.feed_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.feed_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
    return self.x_feeds[ix], self.y_feeds[ix]
end

-- exponential moving average
function StockDataLoader:ema(period)
    local alpha = 2.0/(period + 1) -- decay percentage
    assert(self.row_num == self.stock_data:size(1))
    local ema_data = torch.zeros(self.row_num)
    ema_data[1] = self.stock_data[1][1]
    for i=2, self.row_num do
        ema_data[i] = ema_data[i-1] + alpha * (self.stock_data[i][1] - ema_data[i-1])
    end
    return ema_data
end

-- simple moving average with +/-2*sigma
function StockDataLoader:sma(period)
    local data_size = self.row_num
    local sma_data = torch.Tensor(data_size, 3)
    local multiplier = 1.0/period
    local avg = 0.0
    local avgsq = 0.0
    local head = self.stock_data[1][1]
    local headsq = head * head
    for i=1, period do
        avg = avg + self.stock_data[i][1]
        avgsq = avgsq + self.stock_data[i][1] * self.stock_data[i][1]
    end
    local tmp1
    local tmp2
    for i=(period+1), self.row_num do
        tmp1 = avg * multiplier
        tmp2 = avgsq * multiplier
        tmp2 = math.sqrt(tmp2 - tmp1 * tmp1)
        sma_data[i-period][1] = tmp1
        sma_data[i-period][2] = tmp1 - 2.0 * tmp2
        sma_data[i-period][3] = tmp1 + 2.0 * tmp2
        avg = avg - head + self.stock_data[i][1]
        avgsq = avgsq - headsq + self.stock_data[i][1] * self.stock_data[i][1]
        head = self.stock_data[i-period+1][1]
        headsq = head * head
    end
    return sma_data
end

return StockDataLoader
