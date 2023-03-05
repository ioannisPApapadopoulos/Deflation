% Basic damping of the Newton/active-set update
classdef BasicLinesearch
    methods
        function update = adjust(~, ~, update, damping)
            update = update * damping;
        end 
    end
end

% A backtracking linesearch
% classdef BackTrackingLinesearch
%     
% end
