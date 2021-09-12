#include <vstl/vstlconfig.h>
#include <Common/Common.h>
#include <Common/GameTimer.h>
#include <Windows.h>
uint64_t GameTimer::frameCount = 0;
double GameTimer::deltaTime = 1000;
double GameTimer::time = 0;
GameTimer::GameTimer()
	: mSecondsPerCount(0.0), mDeltaTime(-1.0), mBaseTime(0),
	  mPausedTime(0), mPrevTime(0), mCurrTime(0), mStopped(false) {
	int64_t countsPerSec;
	QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
	mSecondsPerCount = 1.0 / (double)countsPerSec;
}
// Returns the total time elapsed since Reset() was called, NOT counting any
// time when the clock is stopped.
double GameTimer::TotalTime() const {
	// If we are stopped, do not count the time that has passed since we stopped.
	// Moreover, if we previously already had a pause, the distance
	// mStopTime - mBaseTime includes paused time, which we do not want to count.
	// To correct this, we can subtract the paused time from mStopTime:
	//
	//                     |<--paused time-->|
	// ----*---------------*-----------------*------------*------------*------> time
	//  mBaseTime       mStopTime        startTime     mStopTime    mCurrTime
	if (mStopped) {
		return (((mStopTime - mPausedTime) - mBaseTime) * mSecondsPerCount);
	}
	// The distance mCurrTime - mBaseTime includes paused time,
	// which we do not want to count.  To correct this, we can subtract
	// the paused time from mCurrTime:
	//
	//  (mCurrTime - mPausedTime) - mBaseTime
	//
	//                     |<--paused time-->|
	// ----*---------------*-----------------*------------*------> time
	//  mBaseTime       mStopTime        startTime     mCurrTime

	else {
		return (((mCurrTime - mPausedTime) - mBaseTime) * mSecondsPerCount);
	}
}
double GameTimer::DeltaTime() const {
	return mDeltaTime;
}
void GameTimer::Reset() {
	int64_t currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
	mBaseTime = currTime;
	mPrevTime = currTime;
	mStopTime = 0;
	mStopped = false;
}
void GameTimer::Start() {
	int64_t startTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&startTime);
	// Accumulate the time elapsed between stop and start pairs.
	//
	//                     |<-------d------->|
	// ----*---------------*-----------------*------------> time
	//  mBaseTime       mStopTime        startTime
	if (mStopped) {
		mPausedTime += (startTime - mStopTime);
		mPrevTime = startTime;
		mStopTime = 0;
		mStopped = false;
	}
}
void GameTimer::Stop() {
	if (!mStopped) {
		int64_t currTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
		mStopTime = currTime;
		mStopped = true;
	}
}
void GameTimer::UpdateTimeToStatic(uint64_t frameCount) {
	GameTimer::frameCount = frameCount;
	deltaTime = DeltaTime();
	time = TotalTime();
}
void GameTimer::Tick() {
	if (mStopped) {
		mDeltaTime = 0.0;
		return;
	}
	int64_t currTime;
	QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
	mCurrTime = currTime;
	// Time difference between this frame and the previous.
	mDeltaTime = (mCurrTime - mPrevTime) * mSecondsPerCount;
	// Prepare for next frame.
	mPrevTime = mCurrTime;
	// Force nonnegative.  The DXSDK's CDXUTTimer mentions that if the
	// processor goes into a power save mode or we get shuffled to another
	// processor, then mDeltaTime can be negative.
	if (mDeltaTime < 0.0) {
		mDeltaTime = 0.0;
	}
}
