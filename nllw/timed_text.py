from dataclasses import dataclass
from typing import Optional

@dataclass
class TimedText:
    text: Optional[str] = None
    start: Optional[float] = 0
    end: Optional[float] = 0
    
    def duration(self) -> float:
        return self.end - self.start

    def contains_time(self, time: float) -> bool:
        return self.start <= time <= self.end

    def contains_timespan(self, other: 'TimedText') -> bool:
        return self.start <= other.start and self.end >= other.end

    def overlaps_with(self, other: 'TimedText') -> bool:
        return not (self.end <= other.start or other.end <= self.start)

    def is_within(self, other: 'TimedText') -> bool:
        return other.contains_timespan(self)

    def approximate_cut_at(self, cut_time):
        """
        Each word in text is considered to be of duration (end-start)/len(words in text)
        """
        if not self.text or not self.contains_time(cut_time):
            return self, None

        words = self.text.split()
        num_words = len(words)
        if num_words == 0:
            return self, None

        duration_per_word = self.duration() / num_words
        
        cut_word_index = int((cut_time - self.start) / duration_per_word)
        
        if cut_word_index >= num_words:
            cut_word_index = num_words -1
        
        text0 = " ".join(words[:cut_word_index])
        text1 = " ".join(words[cut_word_index:])

        segment0 = TimedText(start=self.start, end=cut_time, text=text0)
        segment1 = TimedText(start=cut_time, end=self.end, text=text1)

        return segment0, segment1