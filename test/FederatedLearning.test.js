// test/FederatedLearning.test.js
// Run with: truffle test
const FederatedLearning = artifacts.require("FederatedLearning");

contract("FederatedLearning", (accounts) => {
  const owner   = accounts[0];
  const client1 = accounts[1];
  const client2 = accounts[2];
  const stranger = accounts[3];

  let fl;

  beforeEach(async () => {
    fl = await FederatedLearning.new({ from: owner });
  });

  // ── Registration ──────────────────────────────────────────────

  it("registers a device and increments deviceCount", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    const count = await fl.deviceCount();
    assert.equal(count.toNumber(), 1, "deviceCount should be 1");
  });

  it("sets initial reputation to 100 on registration", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    const rep = await fl.getDeviceReputation(client1);
    assert.equal(rep.toNumber(), 100, "Initial reputation should be 100");
  });

  it("reports device as registered after registerDevice", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    const isReg = await fl.isDeviceRegistered(client1);
    assert.isTrue(isReg, "Device should be registered");
  });

  it("reverts double registration", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    try {
      await fl.registerDevice("dev_1_dup", { from: client1 });
      assert.fail("Should have reverted");
    } catch (err) {
      assert.include(err.message, "already registered");
    }
  });

  it("unregistered address is not registered", async () => {
    const isReg = await fl.isDeviceRegistered(stranger);
    assert.isFalse(isReg, "Stranger should not be registered");
  });

  // ── Model Update Submission ───────────────────────────────────

  it("allows a registered device to submit a model update", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.submitModelUpdate("abc123hash", 75, { from: client1 });
    const count = await fl.getRoundUpdateCount(0);
    assert.equal(count.toNumber(), 1, "Round 0 should have 1 update");
  });

  it("reverts model update from unregistered device", async () => {
    try {
      await fl.submitModelUpdate("abc123hash", 75, { from: stranger });
      assert.fail("Should have reverted");
    } catch (err) {
      assert.include(err.message, "not registered");
    }
  });

  it("increments totalContributions on submission", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.submitModelUpdate("hash1", 70, { from: client1 });
    await fl.submitModelUpdate("hash2", 72, { from: client1 });
    const dev = await fl.devices(client1);
    assert.equal(dev.totalContributions.toNumber(), 2);
  });

  // ── Validation & Reputation ───────────────────────────────────

  it("increases reputation by 10 on passed validation", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.submitModelUpdate("hash1", 80, { from: client1 });
    await fl.validateUpdate(client1, 0, true, { from: owner });
    const rep = await fl.getDeviceReputation(client1);
    assert.equal(rep.toNumber(), 110, "Reputation should be 110 after pass");
  });

  it("decreases reputation by 20 on failed validation", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.submitModelUpdate("hash1", 80, { from: client1 });
    await fl.validateUpdate(client1, 0, false, { from: owner });
    const rep = await fl.getDeviceReputation(client1);
    assert.equal(rep.toNumber(), 80, "Reputation should be 80 after fail");
  });

  it("does not exceed maximum reputation of 200", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    // Submit and pass validation 11 times (+10 each = 110 over 100 = 210, capped at 200)
    for (let i = 0; i < 11; i++) {
      await fl.submitModelUpdate(`hash${i}`, 80, { from: client1 });
      await fl.validateUpdate(client1, 0, true, { from: owner });
    }
    const rep = await fl.getDeviceReputation(client1);
    assert.isAtMost(rep.toNumber(), 200, "Reputation should not exceed 200");
  });

  it("does not go below minimum reputation of 10", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    // Fail validation 10 times (-20 each = -200, floored at 10)
    for (let i = 0; i < 10; i++) {
      await fl.submitModelUpdate(`hash${i}`, 0, { from: client1 });
      await fl.validateUpdate(client1, 0, false, { from: owner });
    }
    const rep = await fl.getDeviceReputation(client1);
    assert.isAtLeast(rep.toNumber(), 10, "Reputation should not go below 10");
  });

  it("reverts validateUpdate from non-owner", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.submitModelUpdate("hash1", 80, { from: client1 });
    try {
      await fl.validateUpdate(client1, 0, true, { from: stranger });
      assert.fail("Should have reverted");
    } catch (err) {
      assert.include(err.message, "Only owner");
    }
  });

  // ── Global Model & Round Progression ─────────────────────────

  it("increments currentRound after updateGlobalModel", async () => {
    const before = await fl.currentRound();
    await fl.updateGlobalModel("globalHash1", { from: owner });
    const after = await fl.currentRound();
    assert.equal(after.toNumber(), before.toNumber() + 1);
  });

  it("records the global model hash for the round", async () => {
    await fl.updateGlobalModel("globalHash42", { from: owner });
    const stored = await fl.globalModelHashes(0);
    assert.equal(stored, "globalHash42");
  });

  it("reverts updateGlobalModel from non-owner", async () => {
    try {
      await fl.updateGlobalModel("hash", { from: stranger });
      assert.fail("Should have reverted");
    } catch (err) {
      assert.include(err.message, "Only owner");
    }
  });

  // ── Device List ───────────────────────────────────────────────

  it("returns registered devices list", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.registerDevice("dev_2", { from: client2 });
    const devs = await fl.getRegisteredDevices();
    assert.equal(devs.length, 2);
    assert.include(devs, client1);
    assert.include(devs, client2);
  });

  // ── Events ───────────────────────────────────────────────────

  it("emits DeviceRegistered event", async () => {
    const tx = await fl.registerDevice("dev_1", { from: client1 });
    const ev = tx.logs.find(l => l.event === "DeviceRegistered");
    assert.ok(ev, "DeviceRegistered event should be emitted");
    assert.equal(ev.args.deviceId, "dev_1");
  });

  it("emits ModelUpdateSubmitted event", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    const tx = await fl.submitModelUpdate("myhash", 77, { from: client1 });
    const ev = tx.logs.find(l => l.event === "ModelUpdateSubmitted");
    assert.ok(ev, "ModelUpdateSubmitted event should be emitted");
    assert.equal(ev.args.modelHash, "myhash");
  });

  it("emits ReputationUpdated event on validation", async () => {
    await fl.registerDevice("dev_1", { from: client1 });
    await fl.submitModelUpdate("hash1", 80, { from: client1 });
    const tx = await fl.validateUpdate(client1, 0, true, { from: owner });
    const ev = tx.logs.find(l => l.event === "ReputationUpdated");
    assert.ok(ev, "ReputationUpdated event should be emitted");
    assert.equal(ev.args.newReputation.toNumber(), 110);
  });
});
