#pragma once
#include <util/vstlconfig.h>
#include <stdint.h>
class LUISA_DLL GameTimer
{
private:
	static uint64_t frameCount;
	static double deltaTime;
	static double time;
public:
	static uint64_t GetFrameCount() { return frameCount; }
	static double GetDeltaTime() { return deltaTime; }
	static double GetTime() { return time; }
	GameTimer();
	double TotalTime()const; // in seconds
	double DeltaTime()const; // in seconds

	void Reset(); // Call before message loop.
	void Start(); // Call when unpaused.
	void Stop();  // Call when paused.
	void Tick();  // Call every frame.
	void UpdateTimeToStatic(uint64_t frameCount);
private:
	double mSecondsPerCount;
	double mDeltaTime;

	int64_t mBaseTime;
	int64_t mPausedTime;
	int64_t mStopTime;
	int64_t mPrevTime;
	int64_t mCurrTime;

	bool mStopped;
};
