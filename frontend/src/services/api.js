import axios from "axios";


const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 10000,
});


// ==========================================
// JOBS
// ==========================================

export const getJobs = () => {
  return API.get("/jobs/");
};


export const createJob = (job) => {
  return API.post("/jobs/", null, {
    params: {
      name: job.name,

      compute_required: Number(
        job.compute_required
      ),

      memory_required: Number(
        job.memory_required
      ),

      accelerator_a_required: Number(
        job.accelerator_a_required || 0
      ),

      accelerator_b_required: Number(
        job.accelerator_b_required || 0
      ),

      priority: Number(
        job.priority || 1
      ),
    },
  });
};


export const completeJob = (jobId) => {
  return API.post(
    `/jobs/${jobId}/complete`
  );
};


export const cancelJob = (jobId) => {
  return API.post(
    `/jobs/${jobId}/cancel`
  );
};


// ==========================================
// SCHEDULER STATISTICS
// ==========================================

export const getSchedulerStatistics = () => {
  return API.get("/jobs/statistics");
};


// ==========================================
// HISTORICAL ANALYTICS
// ==========================================

export const getHistoricalAnalytics = () => {
  return API.get(
    "/jobs/historical-analytics"
  );
};


// ==========================================
// WORKLOAD GENERATOR
// ==========================================

export const generateWorkload = (
  count,
  pattern,
  seed
) => {
  return API.post(
    "/jobs/workload/generate",
    null,
    {
      params: {
        count: Number(count),

        pattern,

        seed:
          seed === "" || seed == null
            ? undefined
            : Number(seed),
      },
    }
  );
};


// ==========================================
// STRESS TESTING
// ==========================================

export const runStressTest = (
  count,
  pattern,
  seed
) => {
  return API.post(
    "/jobs/stress-test",
    null,
    {
      params: {
        count: Number(count),

        pattern,

        seed:
          seed === "" || seed == null
            ? undefined
            : Number(seed),
      },
    }
  );
};


// ==========================================
// WORKLOAD STATISTICS
// ==========================================

export const getWorkloadStatistics = () => {
  return API.get(
    "/jobs/workload/statistics"
  );
};


// ==========================================
// SCHEDULING POLICY
// ==========================================

export const getSchedulingPolicy = () => {
  return API.get(
    "/jobs/policy"
  );
};


export const setSchedulingPolicy = (
  policy
) => {
  return API.post(
    "/jobs/policy",
    null,
    {
      params: {
        policy,
      },
    }
  );
};


// ==========================================
// RESOURCES
// ==========================================

export const getResources = () => {
  return API.get(
    "/resources/"
  );
};


// ==========================================
// DEADLOCK
// ==========================================

export const getDeadlockStatus = () => {
  return API.get(
    "/resources/deadlock"
  );
};


// ==========================================
// RESOURCE ANALYTICS
// ==========================================

export const getResourceAnalytics = () => {
  return API.get(
    "/analytics/resources"
  );
};


// ==========================================
// HEALTH CHECK
// ==========================================

export const checkBackend = () => {
  return API.get(
    "/health"
  );
};


export const checkAPI = () => {
  return API.get(
    "/"
  );
};


// ==========================================
// AXIOS ERROR HANDLER
// ==========================================

API.interceptors.response.use(
  (response) => {
    return response;
  },

  (error) => {
    console.error(
      "API Error:",
      error?.response?.data ||
        error.message
    );

    return Promise.reject(error);
  }
);


export default API;