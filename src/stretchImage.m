function resizedImage = stretchImage(doPlot, img)
%% Function - stretch the image/digit to fill box
% img - a matrix (e.g. 20x20 MNIST handwritten)

ORIG_SIZE = size(img,1);
DIGIT_MARKER_COLOR = 0.5; %value that indicates a portion of the digit was found (NOTE: normalized, 0..1)
NUM_COLUMNS = ORIG_SIZE;
NUM_ROWS = NUM_COLUMNS;

%go columnwise
leftFound = false;
for col = 1 : NUM_COLUMNS
    columnSum = sum(img(:,col));
    %find left side of digit
    if((columnSum)>DIGIT_MARKER_COLOR)
        if(~leftFound)
            newLeftStart = col;
            leftFound = true;
            continue;
        end
    end
    %find right side of digit
    if((leftFound && columnSum==0) || col==NUM_COLUMNS) %special case where digit already extends to edge
        if(col==NUM_COLUMNS)
            newRightStart = col;
        else
            newRightStart = col-1;
        end
        break;
    end
end

imgTransposed = transpose(img); %so we do all columnwise ops

%go row wise
topFound = false;
for row = 1 : NUM_ROWS
    rowSum = sum(imgTransposed(:,row)); %NOTE: row==column in this case. A little confusing.
    %find top side of digit
    if((rowSum)>DIGIT_MARKER_COLOR)
        if(~topFound)
            newTopStart = row;
            topFound = true;
            continue;
        end
    end
    %find bottom side of digit
    if((topFound && rowSum==0) || row==NUM_ROWS) %special case where digit already extends to edge
        if(row == NUM_ROWS)
            newBottomStart = row;
        else
            newBottomStart = row-1;
        end
        break;
    end
end

boundedImage = img(newTopStart:newBottomStart,newLeftStart:newRightStart);
boundedWidth = newRightStart - newLeftStart;
boundedHeight = newBottomStart - newTopStart;
boundedAspectRatio = boundedWidth / boundedHeight;

resizedImage = imresize(boundedImage,[20,20]);

%plot bounded image
if(doPlot)
    figure(500);
    subplot(1,3,1);
    colormap('gray');
    imagesc(img);
    title('Orig');
    %colorbar;
    subplot(1,3,2);
    colormap('gray');
    imagesc(boundedImage);
    %colorbar;
    title(sprintf('Bounded - H %d W %d pixels Ratio %.2f',boundedHeight,boundedWidth,boundedAspectRatio));
    subplot(1,3,3);
    imagesc(resizedImage);
    title(sprintf('Bounded Resized to OrigSize'));
end

    